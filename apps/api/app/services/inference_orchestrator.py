from dataclasses import dataclass

from app.domain.enums import InferenceMode
from app.inference.base import InferenceAdapter, InferenceImage, InferenceOutput, InferenceRequest, InferenceUnavailable
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
    images: tuple[InferenceImage, ...] = (),
    policy: DecisionPolicy | None = None,
) -> OrchestratedInference:
    mode = InferenceMode.FULL if input_complete else InferenceMode.TWO_D_ONLY
    try:
        output = adapter.predict(
            InferenceRequest(
                event_uuid=event_uuid,
                scenario=scenario,
                input_complete=input_complete,
                images=images,
            )
        )
    except InferenceUnavailable:
        unavailable_output = InferenceOutput(
            model_version=getattr(adapter, "model_version", "unavailable"),
            normal_confidence=0.0,
            defect_score=0.0,
            defect_code=None,
            detections=(),
            latency_ms=0,
        )
        return OrchestratedInference(
            "REVIEW", 0.0, None, "MODEL_UNAVAILABLE", mode, unavailable_output
        )
    result = decide(
        InferenceEvidence(
            identity_complete=identity_complete, input_complete=input_complete, inference_mode=mode,
            normal_confidence=output.normal_confidence, defect_score=output.defect_score,
            defect_code=output.defect_code, three_d_hard_fail=three_d_hard_fail,
        ),
        policy or DecisionPolicy(),
    )
    confidence = {
        "FAIL": output.defect_score,
        "PASS": output.normal_confidence,
        "REVIEW": max(output.normal_confidence, output.defect_score),
    }[result.decision.value]
    return OrchestratedInference(
        result.decision.value,
        confidence,
        result.defect_code,
        result.reason_code,
        mode,
        output,
    )
