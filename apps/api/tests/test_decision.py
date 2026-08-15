from app.domain.enums import Decision, InferenceMode
from app.services.decision import DecisionPolicy, InferenceEvidence, decide


def evidence(**overrides: object) -> InferenceEvidence:
    values: dict[str, object] = {
        "identity_complete": True,
        "input_complete": True,
        "inference_mode": InferenceMode.FULL,
        "normal_confidence": 0.99,
        "defect_score": 0.01,
        "defect_code": None,
        "three_d_hard_fail": False,
    }
    values.update(overrides)
    return InferenceEvidence(**values)


def test_incomplete_identity_never_auto_passes() -> None:
    result = decide(evidence(identity_complete=False), DecisionPolicy())

    assert result.decision is Decision.REVIEW
    assert result.reason_code == "IDENTITY_INCOMPLETE"


def test_missing_required_input_never_auto_passes() -> None:
    result = decide(evidence(input_complete=False), DecisionPolicy())

    assert result.decision is Decision.REVIEW
    assert result.reason_code == "INPUT_INCOMPLETE"


def test_three_d_hard_rule_overrides_normal_image_confidence() -> None:
    result = decide(evidence(three_d_hard_fail=True), DecisionPolicy())

    assert result.decision is Decision.FAIL
    assert result.reason_code == "THREE_D_HARD_LIMIT"


def test_strong_defect_evidence_fails_with_defect_code() -> None:
    result = decide(
        evidence(defect_score=0.87, defect_code="BALL_BRIDGE"),
        DecisionPolicy(fail_min_score=0.75),
    )

    assert result.decision is Decision.FAIL
    assert result.defect_code == "BALL_BRIDGE"


def test_complete_high_confidence_normal_input_can_pass() -> None:
    result = decide(
        evidence(normal_confidence=0.985),
        DecisionPolicy(pass_min_confidence=0.97),
    )

    assert result.decision is Decision.PASS
    assert result.reason_code == "POLICY_AUTO_PASS"


def test_uncertain_result_routes_to_review() -> None:
    result = decide(
        evidence(normal_confidence=0.82, defect_score=0.31),
        DecisionPolicy(),
    )

    assert result.decision is Decision.REVIEW
    assert result.reason_code == "LOW_CONFIDENCE"
