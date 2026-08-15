from dataclasses import dataclass

from app.domain.enums import InferenceMode
from app.inference.base import InferenceAdapter, InferenceRequest
from app.services.decision import DecisionPolicy, InferenceEvidence, decide


@dataclass(frozen=True, slots=True)
class OrchestratedInference:
    decision: str
    confidence: float
    defect_code: str | None
    reason_code: str
    mode: InferenceMode
    output: object


def run_inference(
    adapter: InferenceAdapter,
    *,
    event_uuid: str,
    scenario: str,
    input_complete: bool,
    identity_complete: bool = True,
    three_d_hard_fail: bool = False,
    policy: DecisionPolicy | None = None,
) -> OrchestratedInference:
    output = adapter.predict(InferenceRequest(event_uuid=event_uuid, scenario=scenario, input_complete=input_complete))
    mode = InferenceMode.FULL if input_complete else InferenceMode.TWO_D_ONLY
    result = decide(
        InferenceEvidence(
            identity_complete=identity_complete, input_complete=input_complete, inference_mode=mode,
            normal_confidence=output.normal_confidence, defect_score=output.defect_score,
            defect_code=output.defect_code, three_d_hard_fail=three_d_hard_fail,
        ),
        policy or DecisionPolicy(),
    )
    return OrchestratedInference(result.decision.value, output.normal_confidence, result.defect_code, result.reason_code, mode, output)

