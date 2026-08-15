from dataclasses import dataclass
from typing import Protocol


class InferenceUnavailable(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class InferenceRequest:
    event_uuid: str
    scenario: str
    input_complete: bool


@dataclass(frozen=True, slots=True)
class InferenceOutput:
    model_version: str
    normal_confidence: float
    defect_score: float
    defect_code: str | None
    boxes: tuple[tuple[int, int, int, int], ...]
    latency_ms: int


class InferenceAdapter(Protocol):
    def predict(self, request: InferenceRequest) -> InferenceOutput: ...
