from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
from threading import Lock
from time import perf_counter_ns
from typing import Any
from uuid import uuid4

import httpx


LIGHTS = ("R", "G", "B", "RING")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_manifest_file(manifest_path: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate.resolve()
    project_root = manifest_path.resolve().parents[3]
    project_path = (project_root / candidate).resolve()
    if project_path.is_file():
        return project_path
    parts = candidate.parts
    fixture_prefix = ("data", "external", "pcb_stability_samples")
    if tuple(parts[:3]) == fixture_prefix:
        mounted_path = (manifest_path.resolve().parent / Path(*parts[3:])).resolve()
        if mounted_path.is_file():
            return mounted_path
    raise FileNotFoundError(f"Manifest image is unavailable: {candidate.name}")


def _payload(
    sequence: str,
    path: Path,
    *,
    scenario: str = "REVIEW",
    station: str,
    run_id: str,
) -> dict[str, object]:
    file_hash = _file_hash(path)
    return {
        "DeviceID": "PIS-PCB-SIM",
        "DeviceSessionID": f"PCB-E2E-{run_id}",
        "InspectionSequence": sequence,
        "ProductID": "PCB-STABILITY",
        "BatchID": f"PCB-E2E-{run_id}",
        "TrayID": f"PCB-{run_id}",
        "SlotIndex": sequence[-8:],
        "Station": station,
        "Surface": "TOP",
        "Scenario": scenario,
        "Images": [
            {"LightGroup": light, "Path": str(path), "SHA256": file_hash}
            for light in LIGHTS
        ],
    }


def _nearest_rank(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return round(ordered[index], 3)


def run_e2e(
    client: httpx.Client,
    *,
    api_url: str,
    manifest_path: Path,
    report_path: Path,
    run_id: str | None = None,
    loops: int = 1,
    concurrency: int = 1,
) -> dict[str, object]:
    if loops < 1:
        raise ValueError("loops must be at least 1")
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    namespace = (run_id or uuid4().hex[:8]).upper()[:8]
    sample_station = f"ST-PCB-{namespace}"
    alert_station = f"ST-PCB-ALERT-{namespace}"
    report: dict[str, Any] = {
        "synthetic_decision": True,
        "run_id": namespace,
        "started_at": _utc_now(),
        "finished_at": None,
        "counts": {"requests": 0, "created": 0, "idempotent": 0, "quarantined": 0, "failed": 0},
        "latency_ms": {"p50": 0.0, "p95": 0.0},
        "events": [],
        "quarantines": [],
        "alert_id": None,
        "report_id": None,
        "assertions": [],
    }
    latencies: list[float] = []
    report_lock = Lock()

    def assertion(name: str, passed: bool, detail: str = "") -> None:
        with report_lock:
            report["assertions"].append({"name": name, "passed": passed, "detail": detail})

    def request(method: str, path: str, **kwargs: object) -> httpx.Response:
        with report_lock:
            report["counts"]["requests"] += 1
        started = perf_counter_ns()
        try:
            return client.request(method, f"{api_url.rstrip('/')}/{path.lstrip('/')}", **kwargs)
        except Exception:
            with report_lock:
                report["counts"]["failed"] += 1
            raise
        finally:
            with report_lock:
                latencies.append((perf_counter_ns() - started) / 1_000_000)

    def expect_status(name: str, response: httpx.Response, expected_status: int) -> bool:
        passed = response.status_code == expected_status
        assertion(name, passed, f"HTTP {response.status_code}")
        if not passed:
            with report_lock:
                report["counts"]["failed"] += 1
        return passed

    def import_event(payload: dict[str, object], expected_status: int, name: str) -> dict[str, object]:
        response = request("POST", "/inspections/import/pis-in", json=payload)
        body = response.json()
        expect_status(name, response, expected_status)
        if response.status_code == 201:
            with report_lock:
                report["counts"]["created"] += 1
                report["events"].append(body.get("event_uuid"))
        elif response.status_code == 200:
            with report_lock:
                report["counts"]["idempotent"] += 1
        elif response.status_code == 202:
            with report_lock:
                report["counts"]["quarantined"] += 1
                report["quarantines"].append(
                    {"quarantine_id": body.get("quarantine_id"), "reason_code": body.get("reason_code")}
                )
        return body

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        image_paths = [_resolve_manifest_file(manifest_path, item["normalized_file"]) for item in manifest]
        first_path, second_path = image_paths[0], image_paths[1]

        baseline = _payload(f"{namespace}-SAMPLE-00-00", first_path, station=sample_station, run_id=namespace)
        first = import_event(baseline, 201, "valid image import")
        replay = import_event(baseline, 200, "exact idempotent replay")
        shuffled = {**baseline, "Images": list(reversed(baseline["Images"]))}
        shuffled_replay = import_event(shuffled, 200, "shuffled light replay")
        assertion(
            "replays preserve event UUID",
            first.get("event_uuid") == replay.get("event_uuid") == shuffled_replay.get("event_uuid"),
        )
        detail = request("GET", f"/inspections/{first.get('event_uuid')}")
        expect_status("baseline detail available", detail, 200)
        detail_body = detail.json()
        assertion(
            "replay keeps one event evidence set",
            detail_body.get("attachment_count") == 4 and detail_body.get("inference_result_count") == 1,
        )

        conflict = json.loads(json.dumps(baseline))
        conflict["Images"][0] = {
            "LightGroup": "R",
            "Path": str(second_path),
            "SHA256": _file_hash(second_path),
        }
        body = import_event(conflict, 202, "idempotency conflict quarantined")
        assertion("conflict reason", body.get("reason_code") == "IDEMPOTENCY_CONFLICT")

        missing = _payload(f"{namespace}-MISSING-LIGHT", first_path, station=sample_station, run_id=namespace)
        missing["Images"] = missing["Images"][:3]
        body = import_event(missing, 202, "missing light quarantined")
        assertion("missing light reason", body.get("reason_code") == "LIGHT_SET_INVALID")

        bad_hash = _payload(f"{namespace}-BAD-HASH", first_path, station=sample_station, run_id=namespace)
        bad_hash["Images"][0]["SHA256"] = "0" * 64
        body = import_event(bad_hash, 202, "bad hash quarantined")
        assertion("bad hash reason", body.get("reason_code") == "HASH_MISMATCH")

        corrupt_path = first_path.parent / "invalid_truncated.jpg"
        body = import_event(
            _payload(f"{namespace}-CORRUPT", corrupt_path, station=sample_station, run_id=namespace),
            202,
            "corrupt image quarantined",
        )
        assertion("corrupt image reason", body.get("reason_code") == "IMAGE_DECODE_FAILED")

        outside_path = manifest_path.resolve()
        outside = _payload(f"{namespace}-OUTSIDE", first_path, station=sample_station, run_id=namespace)
        outside_hash = _file_hash(outside_path)
        outside["Images"] = [
            {"LightGroup": light, "Path": str(outside_path), "SHA256": outside_hash}
            for light in LIGHTS
        ]
        body = import_event(outside, 202, "outside-root path quarantined")
        assertion("outside-root reason", body.get("reason_code") == "PATH_OUTSIDE_ROOT")

        reviewed = request(
            "POST",
            "/reviews",
            json={
                "event_uuid": first.get("event_uuid"),
                "decision": "FAIL",
                "defect_code": "PCB_SYNTHETIC",
                "comment": "Synthetic PCB end-to-end review",
                "reviewer": "pcb_e2e",
            },
        )
        expect_status("manual review accepted", reviewed, 201)
        assertion("manual review persisted", reviewed.json().get("golden_status") == "CONFIRMED")
        review_queue = request("GET", "/reviews")
        expect_status("review queue available", review_queue, 200)
        queued_ids = {item.get("event_uuid") for item in review_queue.json().get("items", [])}
        assertion("reviewed event leaves queue", first.get("event_uuid") not in queued_ids)
        reviewed_detail = request("GET", f"/inspections/{first.get('event_uuid')}")
        expect_status("reviewed event detail available", reviewed_detail, 200)
        reviewed_body = reviewed_detail.json()
        assertion(
            "manual review updates event decision",
            reviewed_body.get("decision") == "FAIL" and reviewed_body.get("reason_code") == "MANUAL_REVIEW",
        )

        sample_work = [
            (loop_index, image_index, image_path)
            for loop_index in range(loops)
            for image_index, image_path in enumerate(image_paths)
            if not (loop_index == 0 and image_index == 0)
        ]

        def import_sample(item: tuple[int, int, Path]) -> None:
            loop_index, image_index, image_path = item
            sequence = f"{namespace}-SAMPLE-{loop_index:02d}-{image_index:02d}"
            import_event(
                _payload(sequence, image_path, station=sample_station, run_id=namespace),
                201,
                f"sample {loop_index:02d}-{image_index:02d}",
            )

        if concurrency == 1:
            for item in sample_work:
                import_sample(item)
        else:
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                list(executor.map(import_sample, sample_work))

        alert_fail_events: list[str] = []
        for index in range(20):
            scenario = "DEFECT" if index < 2 else "NORMAL"
            path = image_paths[index % len(image_paths)]
            body = import_event(
                _payload(
                    f"{namespace}-ALERT-{index:03d}",
                    path,
                    scenario=scenario,
                    station=alert_station,
                    run_id=namespace,
                ),
                201,
                f"alert event {index:03d}",
            )
            if scenario == "DEFECT" and body.get("event_uuid"):
                alert_fail_events.append(str(body["event_uuid"]))

        alerts = request("GET", "/alerts").json().get("items", [])
        alert = next((item for item in alerts if item.get("station") == alert_station), None)
        assertion(
            "10 percent alert opened",
            alert is not None
            and alert.get("defect_rate") == 0.1
            and alert.get("sample_count") == 20
            and alert.get("status") == "OPEN",
        )
        if alert:
            report["alert_id"] = alert["alert_id"]
            acknowledged = request(
                "POST", f"/alerts/{alert['alert_id']}/acknowledge", json={"operator": "pcb_e2e"}
            )
            expect_status("alert acknowledgement accepted", acknowledged, 200)
            assertion("alert acknowledged", acknowledged.json().get("status") == "ACKNOWLEDGED")
            created_report = request("POST", "/reports", json={"alert_id": alert["alert_id"]})
            created_body = created_report.json()
            report["report_id"] = created_body.get("report_id")
            expect_status("draft report accepted", created_report, 201)
            assertion("draft report created", created_body.get("status") == "DRAFT")
            if report["report_id"]:
                detail = request("GET", f"/reports/{report['report_id']}")
                report_events = set(detail.json().get("event_uuids", []))
                expect_status("draft report detail available", detail, 200)
                assertion(
                    "report references two synthetic failures",
                    detail.status_code == 200 and report_events == set(alert_fail_events),
                )

        if first.get("event_uuid"):
            preview = request("GET", f"/inspections/{first['event_uuid']}/image")
            expect_status("verified image preview available", preview, 200)
            assertion("preview matches stored source bytes", sha256(preview.content).hexdigest() == _file_hash(first_path))
    except Exception as exc:
        assertion("workflow completed without exception", False, f"{type(exc).__name__}: {exc}")
    finally:
        report["finished_at"] = _utc_now()
        report["latency_ms"] = {
            "p50": _nearest_rank(latencies, 0.50),
            "p95": _nearest_rank(latencies, 0.95),
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> int:
    api_url = os.getenv("API_URL", "http://api:8000/api/v1")
    manifest_path = Path(os.getenv("SIM_MANIFEST_PATH", "/aoi-images/manifest.json"))
    report_path = Path(os.getenv("SIM_REPORT_PATH", "/sim-output/pcb-e2e-report.json"))
    run_id = os.getenv("SIM_RUN_ID")
    loops = int(os.getenv("SIM_LOOPS", "1"))
    concurrency = int(os.getenv("SIM_CONCURRENCY", "1"))
    with httpx.Client(timeout=15) as client:
        report = run_e2e(
            client,
            api_url=api_url,
            manifest_path=manifest_path,
            report_path=report_path,
            run_id=run_id,
            loops=loops,
            concurrency=concurrency,
        )
    return 0 if report["counts"]["failed"] == 0 and all(item["passed"] for item in report["assertions"]) else 1
