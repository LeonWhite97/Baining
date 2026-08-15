from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
from threading import Barrier
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from app.main import create_app
from app.inference.base import Detection, InferenceOutput, InferenceRequest
from app.inference.factory import UnavailableInferenceAdapter
from app.models import Attachment, InferenceResult, InspectionEvent, QuarantineEvent
from app.api.routes import operations
from tests.image_fixtures import make_pis_in_payload, write_image


@pytest.fixture
def client() -> Iterator[TestClient]:
    app = create_app(database_url="sqlite+pysqlite:///:memory:", mode="demo")
    with TestClient(app) as test_client:
        response = test_client.post("/api/v1/demo/reset", json={"seed": 202408})
        assert response.status_code == 200
        yield test_client


@pytest.fixture
def image_case() -> Iterator[tuple[TestClient, object, Path]]:
    image_root = Path(__file__).parents[3] / "tmp" / "image-api-cases" / uuid4().hex
    image_root.mkdir(parents=True)
    app = create_app(database_url="sqlite+pysqlite:///:memory:", mode="demo", image_root=image_root)
    try:
        with TestClient(app) as test_client:
            yield test_client, app, image_root
    finally:
        shutil.rmtree(image_root, ignore_errors=True)


def test_demo_reset_builds_deterministic_dashboard(client: TestClient) -> None:
    response = client.get("/api/v1/dashboard/summary")

    assert response.status_code == 200
    assert response.json()["counts"] == {
        "total": 24,
        "pass": 16,
        "fail": 4,
        "review": 4,
    }
    assert response.json()["open_alerts"] == 1
    assert len(response.json()["defect_trend"]) == 12


def test_tray_map_returns_traceable_slots(client: TestClient) -> None:
    response = client.get("/api/v1/trays/TRAY-001")

    assert response.status_code == 200
    payload = response.json()
    assert payload["tray_id"] == "TRAY-001"
    assert len(payload["slots"]) == 12
    assert payload["slots"][0]["event_uuid"]
    assert payload["slots"][0]["slot_index"] == "01"


def test_review_decision_persists_and_leaves_queue(client: TestClient) -> None:
    queue = client.get("/api/v1/reviews").json()["items"]
    assert len(queue) == 4

    response = client.post(
        "/api/v1/reviews",
        json={
            "event_uuid": queue[0]["event_uuid"],
            "decision": "FAIL",
            "defect_code": "BALL_BRIDGE",
            "comment": "复核确认桥连",
            "reviewer": "qa_demo",
        },
    )

    assert response.status_code == 201
    assert response.json()["golden_status"] == "CONFIRMED"
    assert len(client.get("/api/v1/reviews").json()["items"]) == 3


def test_alert_acknowledgement_creates_persisted_report(client: TestClient) -> None:
    alert = client.get("/api/v1/alerts").json()["items"][0]
    acknowledged = client.post(
        f"/api/v1/alerts/{alert['alert_id']}/acknowledge",
        json={"operator": "line_leader"},
    )
    assert acknowledged.status_code == 200
    assert acknowledged.json()["status"] == "ACKNOWLEDGED"

    report = client.post("/api/v1/reports", json={"alert_id": alert["alert_id"]})
    assert report.status_code == 201
    assert report.json()["status"] == "DRAFT"
    assert report.json()["agent_status"] == "NOT_CONFIGURED"
    assert report.json()["observed_facts"][0].startswith("工站 ST-02")

    report_id = report.json()["report_id"]
    assert client.get(f"/api/v1/reports/{report_id}").status_code == 200


def test_project_profile_contains_delivery_facts_but_no_budget(client: TestClient) -> None:
    response = client.get("/api/v1/project-profile")

    assert response.status_code == 200
    payload = response.json()
    assert payload["period"] == "2024.09-2025.01"
    assert payload["team_count"] == 8
    assert len(payload["agents"]) == 3
    assert payload["quality_targets"]["controlled_rollout"] == "<=3%"
    assert "budget" not in str(payload).lower()


def test_same_source_identity_is_idempotent(client: TestClient) -> None:
    payload = {
        "device_id": "PIS-01",
        "device_session_id": "BOOT-X",
        "inspection_sequence": "9001",
        "product_id": "BGA-256",
        "batch_id": "LOT-202408",
        "tray_id": "TRAY-X",
        "slot_index": "01",
        "station": "ST-01",
        "surface": "TOP",
        "scenario": "NORMAL",
    }

    first = client.post("/api/v1/inspections", json=payload)
    second = client.post("/api/v1/inspections", json=payload)

    assert first.status_code == 201
    assert second.status_code == 200
    assert first.json()["event_uuid"] == second.json()["event_uuid"]


def test_demo_app_can_seed_a_ready_to_use_environment_on_startup() -> None:
    app = create_app(
        database_url="sqlite+pysqlite:///:memory:",
        mode="demo",
        auto_seed=202408,
    )

    with TestClient(app) as seeded_client:
        summary = seeded_client.get("/api/v1/dashboard/summary").json()

    assert summary["counts"]["total"] == 24
    assert summary["open_alerts"] == 1


def test_pis_in_import_normalizes_attachments_and_is_idempotent(
    image_case: tuple[TestClient, object, Path],
) -> None:
    client, app, image_root = image_case
    image_path = image_root / "board.jpg"
    payload = make_pis_in_payload(image_path, write_image(image_path), scenario="REVIEW")

    first = client.post("/api/v1/inspections/import/pis-in", json=payload)
    payload["Images"] = list(reversed(payload["Images"]))
    second = client.post("/api/v1/inspections/import/pis-in", json=payload)

    assert first.status_code == 201
    assert first.json()["decision"] == "REVIEW"
    assert first.json()["attachment_count"] == 4
    assert second.status_code == 200
    assert second.json()["event_uuid"] == first.json()["event_uuid"]
    with app.state.session_factory() as session:
        assert session.scalar(select(func.count()).select_from(InspectionEvent)) == 1
        assert session.scalar(select(func.count()).select_from(Attachment)) == 4
        assert session.scalar(select(func.count()).select_from(InferenceResult)) == 1

    detail = client.get(f"/api/v1/inspections/{first.json()['event_uuid']}")
    assert detail.status_code == 200
    assert detail.json()["attachment_count"] == 4
    assert detail.json()["inference_result_count"] == 1


def test_pis_in_import_quarantines_missing_identity(
    image_case: tuple[TestClient, object, Path],
) -> None:
    client, app, _ = image_case
    response = client.post("/api/v1/inspections/import/pis-in", json={"DeviceID": "PIS-01"})

    assert response.status_code == 202
    assert response.json()["status"] == "QUARANTINED"
    assert response.json()["reason_code"] == "IDENTITY_MISSING"
    assert response.json()["quarantine_id"].startswith("QRN-")
    with app.state.session_factory() as session:
        quarantine = session.scalar(select(QuarantineEvent))
        assert quarantine is not None
        assert quarantine.parse_error.startswith("IDENTITY_MISSING:")
        assert session.scalar(select(func.count()).select_from(InspectionEvent)) == 0


def test_pis_in_import_respects_attachment_event_foreign_key(
    image_case: tuple[TestClient, object, Path],
) -> None:
    client, app, image_root = image_case
    engine = app.state.session_factory.kw["bind"]
    with engine.connect() as connection:
        connection.exec_driver_sql("PRAGMA foreign_keys=ON")
    image_path = image_root / "fk.jpg"
    payload = make_pis_in_payload(image_path, write_image(image_path), scenario="REVIEW")
    response = client.post("/api/v1/inspections/import/pis-in", json=payload)

    assert response.status_code == 201
    assert response.json()["attachment_count"] == 4


def test_pis_in_import_quarantines_invalid_evidence_without_partial_writes(
    image_case: tuple[TestClient, object, Path],
) -> None:
    client, app, image_root = image_case
    image_path = image_root / "board.jpg"
    payload = make_pis_in_payload(image_path, write_image(image_path))
    payload["Images"][0]["SHA256"] = "0" * 64

    response = client.post("/api/v1/inspections/import/pis-in", json=payload)

    assert response.status_code == 202
    assert response.json()["reason_code"] == "HASH_MISMATCH"
    assert str(image_root) not in response.json()["reason"]
    with app.state.session_factory() as session:
        assert session.scalar(select(func.count()).select_from(QuarantineEvent)) == 1
        assert session.scalar(select(func.count()).select_from(InspectionEvent)) == 0
        assert session.scalar(select(func.count()).select_from(Attachment)) == 0
        assert session.scalar(select(func.count()).select_from(InferenceResult)) == 0


@pytest.mark.parametrize(
    "images",
    ["not-a-list", [{"LightGroup": "R"}], [None, None, None, None]],
)
def test_pis_in_import_quarantines_malformed_attachment_metadata(
    image_case: tuple[TestClient, object, Path],
    images: object,
) -> None:
    client, app, image_root = image_case
    image_path = image_root / "metadata.jpg"
    payload = make_pis_in_payload(image_path, write_image(image_path))
    payload["Images"] = images

    response = client.post("/api/v1/inspections/import/pis-in", json=payload)

    assert response.status_code == 202
    assert response.json()["reason_code"] == "LIGHT_SET_INVALID"
    with app.state.session_factory() as session:
        assert session.scalar(select(func.count()).select_from(InspectionEvent)) == 0


def test_pis_in_import_quarantines_idempotency_conflict_and_keeps_original(
    image_case: tuple[TestClient, object, Path],
) -> None:
    client, app, image_root = image_case
    first_path = image_root / "first.jpg"
    second_path = image_root / "second.jpg"
    payload = make_pis_in_payload(first_path, write_image(first_path))
    first = client.post("/api/v1/inspections/import/pis-in", json=payload)
    second_hash = write_image(second_path, size=(80, 60))
    payload["Images"][0] = {"LightGroup": "R", "Path": str(second_path), "SHA256": second_hash}

    conflict = client.post("/api/v1/inspections/import/pis-in", json=payload)

    assert first.status_code == 201
    assert conflict.status_code == 202
    assert conflict.json()["reason_code"] == "IDEMPOTENCY_CONFLICT"
    with app.state.session_factory() as session:
        assert session.scalar(select(func.count()).select_from(InspectionEvent)) == 1
        assert session.scalar(select(func.count()).select_from(Attachment)) == 4
        stored_paths = set(session.scalars(select(Attachment.file_path)).all())
        assert stored_paths == {str(first_path.resolve())}


def test_pis_in_import_recovers_concurrent_identical_requests() -> None:
    case_dir = Path(__file__).parents[3] / "tmp" / "image-concurrency-cases" / uuid4().hex
    case_dir.mkdir(parents=True)
    database_path = (case_dir / "concurrency.db").as_posix()
    image_path = case_dir / "board.jpg"
    payload = make_pis_in_payload(image_path, write_image(image_path))
    app = create_app(
        database_url=f"sqlite+pysqlite:///{database_path}",
        mode="demo",
        image_root=case_dir,
    )
    try:
        with TestClient(app) as client, ThreadPoolExecutor(max_workers=4) as executor:
            responses = list(executor.map(lambda _: client.post("/api/v1/inspections/import/pis-in", json=payload), range(4)))

        assert sorted(response.status_code for response in responses) == [200, 200, 200, 201]
        assert len({response.json()["event_uuid"] for response in responses}) == 1
        with app.state.session_factory() as session:
            assert session.scalar(select(func.count()).select_from(InspectionEvent)) == 1
            assert session.scalar(select(func.count()).select_from(Attachment)) == 4
            assert session.scalar(select(func.count()).select_from(InferenceResult)) == 1
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)


def test_pis_in_import_quarantines_concurrent_different_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    case_dir = Path(__file__).parents[3] / "tmp" / "image-concurrency-cases" / uuid4().hex
    case_dir.mkdir(parents=True)
    first_path = case_dir / "first.jpg"
    second_path = case_dir / "second.jpg"
    first_payload = make_pis_in_payload(first_path, write_image(first_path))
    second_payload = make_pis_in_payload(second_path, write_image(second_path, size=(80, 60)))
    barrier = Barrier(2)
    original_run_inference = operations.run_inference

    def synchronized_inference(*args: object, **kwargs: object) -> object:
        barrier.wait(timeout=5)
        return original_run_inference(*args, **kwargs)

    monkeypatch.setattr(operations, "run_inference", synchronized_inference)
    app = create_app(
        database_url=f"sqlite+pysqlite:///{(case_dir / 'conflict.db').as_posix()}",
        mode="demo",
        image_root=case_dir,
    )
    try:
        with TestClient(app) as client, ThreadPoolExecutor(max_workers=2) as executor:
            responses = list(
                executor.map(
                    lambda payload: client.post("/api/v1/inspections/import/pis-in", json=payload),
                    (first_payload, second_payload),
                )
            )

        assert sorted(response.status_code for response in responses) == [201, 202]
        quarantined = next(response for response in responses if response.status_code == 202)
        assert quarantined.json()["reason_code"] == "IDEMPOTENCY_CONFLICT"
        with app.state.session_factory() as session:
            assert session.scalar(select(func.count()).select_from(InspectionEvent)) == 1
            assert session.scalar(select(func.count()).select_from(Attachment)) == 4
            assert session.scalar(select(func.count()).select_from(InferenceResult)) == 1
            assert session.scalar(select(func.count()).select_from(QuarantineEvent)) == 1
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)


@pytest.mark.parametrize("mode", ["shadow", "controlled"])
def test_pis_in_import_forces_review_outside_demo(
    mode: str,
) -> None:
    case_dir = Path(__file__).parents[3] / "tmp" / "image-mode-cases" / uuid4().hex
    case_dir.mkdir(parents=True)
    image_path = case_dir / "board.jpg"
    payload = make_pis_in_payload(image_path, write_image(image_path), scenario="NORMAL")
    app = create_app(database_url="sqlite+pysqlite:///:memory:", mode=mode, image_root=case_dir)

    try:
        with TestClient(app) as client:
            response = client.post("/api/v1/inspections/import/pis-in", json=payload)

        assert response.status_code == 201
        assert response.json()["decision"] == "REVIEW"
        assert response.json()["reason_code"] == "MODEL_UNAVAILABLE"
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)


def test_pis_in_import_forwards_images_and_persists_all_detections() -> None:
    case_dir = Path(__file__).parents[3] / "tmp" / "image-adapter-cases" / uuid4().hex
    case_dir.mkdir(parents=True)

    class RecordingAdapter:
        model_version = "recording-v1"

        def __init__(self) -> None:
            self.requests: list[InferenceRequest] = []

        def predict(self, request: InferenceRequest) -> InferenceOutput:
            self.requests.append(request)
            detections = (
                Detection(1, 2, 10, 20, 0, "BALL_BRIDGE", 0.91),
                Detection(4, 5, 10, 20, 6, "FOREIGN_MATERIAL", 0.80),
            )
            return InferenceOutput(
                model_version=self.model_version,
                normal_confidence=0.0,
                defect_score=0.91,
                defect_code="BALL_BRIDGE",
                detections=detections,
                latency_ms=12,
            )

    adapter = RecordingAdapter()
    image_path = case_dir / "board.jpg"
    payload = make_pis_in_payload(image_path, write_image(image_path), scenario="DEFECT")
    app = create_app(
        database_url="sqlite+pysqlite:///:memory:",
        mode="demo",
        image_root=case_dir,
        inference_adapter=adapter,
    )
    try:
        with TestClient(app) as client:
            response = client.post("/api/v1/inspections/import/pis-in", json=payload)

        assert response.status_code == 201
        assert response.json()["defect_code"] == "BALL_BRIDGE"
        assert tuple(image.light_id for image in adapter.requests[0].images) == ("R", "G", "B", "RING")
        assert all(image.path.is_file() for image in adapter.requests[0].images)
        with app.state.session_factory() as session:
            stored = session.scalar(select(InferenceResult))
            assert stored is not None
            assert len(stored.defect_bbox) == 2
            assert set(stored.defect_bbox[0]) == {
                "x", "y", "w", "h", "class_id", "defect_code", "confidence"
            }
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)


def test_unavailable_inference_persists_auditable_review() -> None:
    case_dir = Path(__file__).parents[3] / "tmp" / "image-adapter-cases" / uuid4().hex
    case_dir.mkdir(parents=True)
    image_path = case_dir / "board.jpg"
    payload = make_pis_in_payload(image_path, write_image(image_path), scenario="NORMAL")
    app = create_app(
        database_url="sqlite+pysqlite:///:memory:",
        mode="shadow",
        image_root=case_dir,
        inference_adapter=UnavailableInferenceAdapter("fc-bga-requested-v1"),
    )
    try:
        with TestClient(app) as client:
            response = client.post("/api/v1/inspections/import/pis-in", json=payload)

        assert response.status_code == 201
        body = response.json()
        assert body["decision"] == "REVIEW"
        assert body["reason_code"] == "MODEL_UNAVAILABLE"
        assert body["defect_code"] is None
        assert body["attachment_count"] == 4
        with app.state.session_factory() as session:
            stored = session.scalar(select(InferenceResult))
            assert stored is not None
            assert stored.model_version == "fc-bga-requested-v1"
            assert stored.defect_bbox == []
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)


def test_image_preview_returns_exact_verified_evidence(
    image_case: tuple[TestClient, object, Path],
) -> None:
    client, _, image_root = image_case
    image_path = image_root / "preview.jpg"
    payload = make_pis_in_payload(image_path, write_image(image_path))
    created = client.post("/api/v1/inspections/import/pis-in", json=payload).json()

    default_preview = client.get(f"/api/v1/inspections/{created['event_uuid']}/image")
    explicit_preview = client.get(
        f"/api/v1/inspections/{created['event_uuid']}/image", params={"light_id": "G"}
    )

    assert default_preview.status_code == 200
    assert default_preview.headers["content-type"].startswith("image/jpeg")
    assert default_preview.content == image_path.read_bytes()
    assert explicit_preview.content == image_path.read_bytes()


@pytest.mark.parametrize(
    ("filename", "image_format", "expected_media_type"),
    [("jpeg-as-bin.bin", "JPEG", "image/jpeg"), ("png-as-jpeg.jpg", "PNG", "image/png")],
)
def test_image_preview_uses_verified_content_type(
    image_case: tuple[TestClient, object, Path],
    filename: str,
    image_format: str,
    expected_media_type: str,
) -> None:
    client, _, image_root = image_case
    image_path = image_root / filename
    payload = make_pis_in_payload(
        image_path,
        write_image(image_path, image_format=image_format),
    )
    created = client.post("/api/v1/inspections/import/pis-in", json=payload).json()

    preview = client.get(f"/api/v1/inspections/{created['event_uuid']}/image")

    assert preview.status_code == 200
    assert preview.headers["content-type"].startswith(expected_media_type)
    assert preview.content == image_path.read_bytes()


def test_image_preview_rejects_unknown_or_changed_evidence(
    image_case: tuple[TestClient, object, Path],
) -> None:
    client, _, image_root = image_case
    image_path = image_root / "mutable.jpg"
    payload = make_pis_in_payload(image_path, write_image(image_path))
    event_uuid = client.post("/api/v1/inspections/import/pis-in", json=payload).json()["event_uuid"]

    assert client.get("/api/v1/inspections/missing/image").status_code == 404
    assert client.get(
        f"/api/v1/inspections/{event_uuid}/image", params={"light_id": "SIDE"}
    ).status_code == 404

    write_image(image_path, size=(80, 60))
    assert client.get(f"/api/v1/inspections/{event_uuid}/image").status_code == 409

    image_path.unlink()
    assert client.get(f"/api/v1/inspections/{event_uuid}/image").status_code == 409


def test_simulator_ingestion_triggers_station_alert_after_minimum_window(client: TestClient) -> None:
    for index in range(20):
        response = client.post(
            "/api/v1/inspections",
            json={
                "device_id": "PIS-ALERT", "device_session_id": "BOOT-ALERT", "inspection_sequence": str(index),
                "product_id": "BGA-256", "batch_id": "LOT-ALERT", "tray_id": "TRAY-ALERT",
                "slot_index": f"{index:02d}", "station": "ST-55", "surface": "TOP",
                "scenario": "DEFECT" if index < 2 else "NORMAL",
            },
        )
        assert response.status_code == 201

    alerts = client.get("/api/v1/alerts").json()["items"]
    assert any(item["station"] == "ST-55" and item["defect_rate"] == 0.1 for item in alerts)
