from collections.abc import Iterator

from fastapi.testclient import TestClient

from app.main import create_app


class TimeoutAgentClient:
    def draft_report(self, payload: dict[str, object]) -> dict[str, object]:
        raise TimeoutError("agent timed out")


def test_agent_timeout_degrades_report_but_does_not_break_realtime_api() -> None:
    app = create_app(database_url="sqlite+pysqlite:///:memory:", mode="demo", auto_seed=202408)
    app.state.agent_client = TimeoutAgentClient()

    with TestClient(app) as client:
        alert = client.get("/api/v1/alerts").json()["items"][0]
        client.post(f"/api/v1/alerts/{alert['alert_id']}/acknowledge", json={"operator": "qa"})
        report = client.post("/api/v1/reports", json={"alert_id": alert["alert_id"]})
        dashboard = client.get("/api/v1/dashboard/summary")
        inspection = client.get("/api/v1/inspections")

    assert report.status_code == 201
    assert report.json()["agent_status"] == "UNAVAILABLE"
    assert report.json()["status"] == "DRAFT"
    assert dashboard.status_code == 200
    assert inspection.status_code == 200

