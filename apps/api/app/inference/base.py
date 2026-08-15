from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class InferenceUnavailable(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class InferenceImage:
    light_id: str
    path: Path
    sha256: str
    width: int
    height: int
    media_type: str


@dataclass(frozen=True, slots=True)
class Detection:
    x: int
    y: int
    w: int
    h: int
    class_id: int
    defect_code: str
    confidence: float


@dataclass(frozen=True, slots=True)
class InferenceRequest:
    event_uuid: str
    scenario: str
    input_complete: bool
    images: tuple[InferenceImage, ...] = ()


@dataclass(frozen=True, slots=True)
class InferenceOutput:
    model_version: str
    normal_confidence: float
    defect_score: float
    defect_code: str | None
    detections: tuple[Detection, ...]
    latency_ms: int

    @property
    def boxes(self) -> tuple[tuple[int, int, int, int], ...]:
        return tuple((item.x, item.y, item.w, item.h) for item in self.detections)


class InferenceAdapter(Protocol):
    model_version: str

    def predict(self, request: InferenceRequest) -> InferenceOutput: ...
