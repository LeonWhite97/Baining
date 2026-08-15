from fastapi.testclient import TestClient

from app.main import create_app


def test_health_reports_runtime_mode_and_version() -> None:
    client = TestClient(create_app(database_url="sqlite+pysqlite:///:memory:", mode="demo"))

    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "mode": "demo", "version": "v3.5"}
