from agent_rag.workflows.model_governance import evaluate_release


def test_nonzero_silent_mismatch_blocks_release() -> None:
    result = evaluate_release(
        {
            "recall": 0.995,
            "auto_pass_escape_rate": 0.001,
            "false_positive_rate": 0.02,
            "review_ratio": 0.05,
            "p95_ms": 80,
            "throughput_per_second": 30,
            "backlog_growth": 0.0,
            "silent_mismatch_rate": 0.0001,
            "shadow_difference_rate": 0.01,
            "blind_test_passed": True,
            "rollback_drill_passed": True,
            "approval_metadata": True,
            "critical_defect_escape": False,
        }
    )

    assert result.action == "BLOCK"
    assert "SILENT_MISMATCH" in result.failed_gates
    assert result.approval_required is True

