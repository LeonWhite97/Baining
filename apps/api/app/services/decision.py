from dataclasses import dataclass

from app.domain.enums import Decision, InferenceMode


@dataclass(frozen=True, slots=True)
class InferenceEvidence:
    identity_complete: bool
    input_complete: bool
    inference_mode: InferenceMode
    normal_confidence: float
    defect_score: float
    defect_code: str | None
    three_d_hard_fail: bool


@dataclass(frozen=True, slots=True)
class DecisionPolicy:
    pass_min_confidence: float = 0.97
    fail_min_score: float = 0.75


@dataclass(frozen=True, slots=True)
class DecisionResult:
    decision: Decision
    reason_code: str
    defect_code: str | None = None


def decide(evidence: InferenceEvidence, policy: DecisionPolicy) -> DecisionResult:
    if not evidence.identity_complete:
        return DecisionResult(Decision.REVIEW, "IDENTITY_INCOMPLETE")
    if not evidence.input_complete:
        return DecisionResult(Decision.REVIEW, "INPUT_INCOMPLETE")
    if evidence.three_d_hard_fail:
        return DecisionResult(Decision.FAIL, "THREE_D_HARD_LIMIT", evidence.defect_code)
    if evidence.defect_score >= policy.fail_min_score:
        return DecisionResult(Decision.FAIL, "DEFECT_SCORE", evidence.defect_code)
    if evidence.normal_confidence >= policy.pass_min_confidence:
        return DecisionResult(Decision.PASS, "POLICY_AUTO_PASS")
    return DecisionResult(Decision.REVIEW, "LOW_CONFIDENCE", evidence.defect_code)
