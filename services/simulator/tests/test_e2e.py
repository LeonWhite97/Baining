import json
from pathlib import Path
import shutil
from uuid import uuid4

import httpx

from simulator.e2e import _resolve_manifest_file, run_e2e


class ScenarioTransport(httpx.BaseTransport):
    def __init__(self) -> None:
        self.events: dict[str, dict[str, object]] = {}
        self.baseline_images: list[dict[str, str]] | None = None
        self.alert_events: list[str] = []
        self.reviewed_events: set[str] = set()
        self.detail_requests = 0
        self.review_queue_requests = 0
        self.alert_station = "ST-PCB-ALERT-TEST"
        self.baseline_content = b""

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        payload = json.loads(request.content.decode()) if request.content else {}
        if path.endswith("/inspections/import/pis-in"):
            return self._import_response(request, payload)
        if path.endswith("/reviews") and request.method == "POST":
            event_uuid = str(payload["event_uuid"])
            self.reviewed_events.add(event_uuid)
            self.events[event_uuid]["decision"] = payload["decision"]
            self.events[event_uuid]["reason_code"] = "MANUAL_REVIEW"
            return httpx.Response(201, json={"golden_status": "CONFIRMED"}, request=request)
        if path.endswith("/reviews") and request.method == "GET":
            self.review_queue_requests += 1
            items = [
                event
                for event_uuid, event in self.events.items()
                if event.get("decision") == "REVIEW" and event_uuid not in self.reviewed_events
            ]
            return httpx.Response(200, json={"items": items, "total": len(items)}, request=request)
        if path.endswith("/alerts"):
            return httpx.Response(
                200,
                json={"items": [{"alert_id": "ALT-PCB", "station": self.alert_station, "defect_rate": 0.1, "sample_count": 20, "status": "OPEN"}]},
                request=request,
            )
        if path.endswith("/alerts/ALT-PCB/acknowledge"):
            return httpx.Response(200, json={"alert_id": "ALT-PCB", "status": "ACKNOWLEDGED"}, request=request)
        if path.endswith("/reports") and request.method == "POST":
            return httpx.Response(
                201,
                json={"report_id": "RPT-PCB", "status": "DRAFT", "event_uuids": self.alert_events[:2]},
                request=request,
            )
        if path.endswith("/reports/RPT-PCB"):
            return httpx.Response(
                200,
                json={"report_id": "RPT-PCB", "status": "DRAFT", "event_uuids": self.alert_events[:2]},
                request=request,
            )
        if path.endswith("/image"):
            return httpx.Response(200, content=self.baseline_content, headers={"content-type": "image/jpeg"}, request=request)
        if "/inspections/" in path and request.method == "GET":
            self.detail_requests += 1
            event_uuid = path.rsplit("/", 1)[-1]
            event = self.events.get(event_uuid)
            if event is not None:
                return httpx.Response(
                    200,
                    json={**event, "attachment_count": 4, "inference_result_count": 1},
                    request=request,
                )
        return httpx.Response(404, json={"detail": "not found"}, request=request)

    def _import_response(self, request: httpx.Request, payload: dict[str, object]) -> httpx.Response:
        sequence = str(payload["InspectionSequence"])
        images = payload.get("Images", [])
        if "SAMPLE-00-00" in sequence:
            if self.baseline_images is None:
                self.baseline_images = sorted(images, key=lambda item: item["LightGroup"])
                self.baseline_content = Path(images[0]["Path"]).read_bytes()
                status_code = 201
            elif sorted(images, key=lambda item: item["LightGroup"]) == self.baseline_images:
                status_code = 200
            else:
                return quarantine(request, "IDEMPOTENCY_CONFLICT")
            return self._created(request, f"EVT-{sequence}", status_code, "REVIEW")
        if "MISSING-LIGHT" in sequence:
            return quarantine(request, "LIGHT_SET_INVALID")
        if "BAD-HASH" in sequence:
            return quarantine(request, "HASH_MISMATCH")
        if "CORRUPT" in sequence:
            return quarantine(request, "IMAGE_DECODE_FAILED")
        if "OUTSIDE" in sequence:
            return quarantine(request, "PATH_OUTSIDE_ROOT")
        if "SAMPLE-" in sequence:
            return self._created(request, f"EVT-{sequence}", 201, "REVIEW")
        if "ALERT-" in sequence:
            event_uuid = f"EVT-{sequence}"
            self.alert_station = str(payload["Station"])
            if payload["Scenario"] == "DEFECT":
                self.alert_events.append(event_uuid)
            return self._created(request, event_uuid, 201, "FAIL" if payload["Scenario"] == "DEFECT" else "PASS")
        return httpx.Response(400, json={"detail": "unexpected sequence"}, request=request)

    def _created(self, request: httpx.Request, event_uuid: str, status_code: int, decision: str) -> httpx.Response:
        body = {"event_uuid": event_uuid, "decision": decision, "attachment_count": 4}
        self.events[event_uuid] = body
        return httpx.Response(status_code, json=body, request=request)


def created(request: httpx.Request, event_uuid: str, status_code: int, decision: str) -> httpx.Response:
    return httpx.Response(
        status_code,
        json={"event_uuid": event_uuid, "decision": decision, "attachment_count": 4},
        request=request,
    )


def quarantine(request: httpx.Request, reason_code: str) -> httpx.Response:
    return httpx.Response(
        202,
        json={"status": "QUARANTINED", "quarantine_id": f"QRN-{reason_code}", "reason_code": reason_code},
        request=request,
    )


def test_run_e2e_writes_complete_passing_report() -> None:
    fixture_root = Path(__file__).parents[3] / "data" / "external" / "pcb_stability_samples"
    case_dir = Path(__file__).parents[3] / "tmp" / "simulator-e2e-cases" / uuid4().hex
    case_dir.mkdir(parents=True)
    report_path = case_dir / "report.json"
    try:
        transport = ScenarioTransport()
        with httpx.Client(transport=transport) as client:
            report = run_e2e(
                client,
                api_url="http://test/api/v1",
                manifest_path=fixture_root / "manifest.json",
                report_path=report_path,
                run_id="TEST",
            )

        assert report["synthetic_decision"] is True
        assert report["counts"] == {"requests": 46, "created": 30, "idempotent": 2, "quarantined": 5, "failed": 0}
        assert report["alert_id"] == "ALT-PCB"
        assert report["report_id"] == "RPT-PCB"
        assert len(report["latency_ms"]) == 2
        assert all(item["passed"] for item in report["assertions"])
        assert transport.detail_requests >= 2
        assert transport.review_queue_requests == 1
        assert json.loads(report_path.read_text(encoding="utf-8"))["report_id"] == "RPT-PCB"
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)


def test_run_e2e_uses_run_namespace_for_repeatability() -> None:
    fixture_root = Path(__file__).parents[3] / "data" / "external" / "pcb_stability_samples"
    case_dir = Path(__file__).parents[3] / "tmp" / "simulator-e2e-cases" / uuid4().hex
    case_dir.mkdir(parents=True)
    transport = ScenarioTransport()
    try:
        with httpx.Client(transport=transport) as client:
            first = run_e2e(
                client,
                api_url="http://test/api/v1",
                manifest_path=fixture_root / "manifest.json",
                report_path=case_dir / "first.json",
                run_id="FIRST",
            )
            transport.baseline_images = None
            transport.alert_events.clear()
            second = run_e2e(
                client,
                api_url="http://test/api/v1",
                manifest_path=fixture_root / "manifest.json",
                report_path=case_dir / "second.json",
                run_id="SECOND",
            )

        assert first["counts"]["failed"] == 0
        assert second["counts"]["failed"] == 0
        assert set(first["events"]).isdisjoint(second["events"])
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)


def test_manifest_file_resolution_supports_container_mount_layout() -> None:
    case_dir = Path(__file__).parents[3] / "tmp" / "simulator-e2e-cases" / uuid4().hex
    mounted_root = case_dir / "aoi-images"
    normalized = mounted_root / "normalized_1920x1080"
    normalized.mkdir(parents=True)
    manifest_path = mounted_root / "manifest.json"
    manifest_path.write_text("[]", encoding="utf-8")
    image_path = normalized / "pcb_01_canonscan.jpg"
    image_path.write_bytes(b"fixture")
    try:
        resolved = _resolve_manifest_file(
            manifest_path,
            "./data/external/pcb_stability_samples/normalized_1920x1080/pcb_01_canonscan.jpg",
        )
        assert resolved == image_path.resolve()
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)
