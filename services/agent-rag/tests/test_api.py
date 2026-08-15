from fastapi.testclient import TestClient

from agent_rag.main import create_app


def test_agent_api_exposes_health_retrieval_and_guardrailed_workflows() -> None:
    client = TestClient(create_app())

    assert client.get("/agent-api/v1/health").json()["status"] == "ok"
    search = client.post(
        "/agent-api/v1/knowledge/search",
        json={"query": "球高超限", "categories": ["SOP"], "limit": 2},
    )
    assert search.status_code == 200
    assert search.json()["items"][0]["citation"]

    quality = client.post(
        "/agent-api/v1/assess-data-quality",
        json={"identity_complete": False, "attachments": []},
    )
    assert quality.json()["risk_level"] == "CRITICAL"


def test_report_and_release_endpoints_require_human_approval() -> None:
    client = TestClient(create_app())
    report = client.post(
        "/agent-api/v1/draft-report",
        json={"station": "ST-02", "defect_rate": 0.25, "threshold": 0.1, "sample_count": 24},
    )
    release = client.post(
        "/agent-api/v1/recommend-model-release",
        json={"silent_mismatch_rate": 0.001},
    )

    assert report.json()["status"] == "DRAFT"
    assert release.json()["approval_required"] is True
    assert release.json()["action"] == "BLOCK"

