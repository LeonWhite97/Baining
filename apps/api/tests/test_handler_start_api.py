from fastapi.testclient import TestClient

from app.main import create_app


PATH = "/api/v1/inspections/aoi/start"


def valid_start_payload(
    *,
    trace_id: str = "TRACE-1",
    cycle_id: str = "CYCLE-1",
    handler_id: str = "HANDLER-1",
    product_id: str = "BGA-256",
) -> dict[str, str]:
    return {
        "trace_id": trace_id,
        "handler_id": handler_id,
        "handler_session_id": "SESSION-1",
        "cycle_id": cycle_id,
        "station_code": "AOI",
        "product_id": product_id,
        "batch_id": "LOT-1",
        "tray_id": "TRAY-1",
        "slot_index": "01",
        "surface": "TOP",
    }


def test_valid_shadow_start_creates_durable_empty_inference_cycle() -> None:
    app = create_app(
        database_url="sqlite+pysqlite:///:memory:", mode="shadow", handler_integration_enabled=True
    )

    with TestClient(app) as client:
        response = client.post(PATH, json=valid_start_payload())

    assert response.status_code == 201
    assert response.json()["trace_id"] == "TRACE-1"
    assert response.json()["association_status"] == "START_RECEIVED"
    assert response.json()["ai_decision"] is None
    assert response.json()["handler_publish_status"] == "NOT_READY"


def test_identical_start_is_idempotent() -> None:
    app = create_app(
        database_url="sqlite+pysqlite:///:memory:", mode="shadow", handler_integration_enabled=True
    )

    with TestClient(app) as client:
        first = client.post(PATH, json=valid_start_payload())
        second = client.post(PATH, json=valid_start_payload())

    assert first.status_code == 201
    assert second.status_code == 200
    assert first.json()["event_uuid"] == second.json()["event_uuid"]


def test_different_trace_is_rejected_while_cycle_active() -> None:
    app = create_app(
        database_url="sqlite+pysqlite:///:memory:", mode="shadow", handler_integration_enabled=True
    )

    with TestClient(app) as client:
        client.post(PATH, json=valid_start_payload())
        response = client.post(PATH, json=valid_start_payload(trace_id="TRACE-2", cycle_id="CYCLE-2"))

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "ACTIVE_CYCLE_EXISTS"


def test_same_trace_with_different_evidence_is_rejected() -> None:
    app = create_app(
        database_url="sqlite+pysqlite:///:memory:", mode="shadow", handler_integration_enabled=True
    )

    with TestClient(app) as client:
        client.post(PATH, json=valid_start_payload())
        response = client.post(PATH, json=valid_start_payload(product_id="BGA-512"))

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "EVIDENCE_CONFLICT"


def test_shadow_start_rejects_scenario_field() -> None:
    app = create_app(
        database_url="sqlite+pysqlite:///:memory:", mode="shadow", handler_integration_enabled=True
    )
    payload = valid_start_payload()
    payload["scenario"] = "NORMAL"

    with TestClient(app) as client:
        response = client.post(PATH, json=payload)

    assert response.status_code == 422


def test_handler_start_route_is_hidden_in_demo_mode() -> None:
    app = create_app(database_url="sqlite+pysqlite:///:memory:", mode="demo")

    with TestClient(app) as client:
        response = client.post(PATH, json=valid_start_payload())

    assert response.status_code == 404


def test_handler_start_route_is_hidden_by_default_in_shadow_mode() -> None:
    app = create_app(database_url="sqlite+pysqlite:///:memory:", mode="shadow")

    with TestClient(app) as client:
        response = client.post(PATH, json=valid_start_payload())

    assert response.status_code == 404
