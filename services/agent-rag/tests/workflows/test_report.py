from agent_rag.workflows.report import run_report_workflow


def test_report_draft_separates_facts_from_hypotheses() -> None:
    draft = run_report_workflow(
        {
            "station": "ST-02",
            "defect_rate": 0.25,
            "threshold": 0.10,
            "sample_count": 24,
            "defect_code": "BALL_BRIDGE",
            "event_uuids": ["evt-1"],
        }
    )

    assert draft.observed_facts
    assert draft.open_questions
    assert all(ref.document_id for ref in draft.evidence_refs)
    assert draft.status == "DRAFT"
