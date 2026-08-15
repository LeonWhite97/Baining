from agent_rag.workflows.data_quality import run_data_quality_workflow


def test_missing_identity_never_recommends_pass() -> None:
    result = run_data_quality_workflow({"identity_complete": False, "attachments": []})

    assert result.risk_level == "CRITICAL"
    assert "AUTO_PASS" not in result.recommended_actions
    assert result.evidence_refs

