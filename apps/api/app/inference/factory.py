from __future__ import annotations

from app.config import InferenceBackend, InferenceSettings, RuntimeMode
from app.inference.base import InferenceOutput, InferenceRequest, InferenceUnavailable
from app.inference.demo import DemoInferenceAdapter
from app.inference.tensorrt import TensorRtInferenceAdapter
from app.inference.ultralytics import UltralyticsInferenceAdapter


class UnavailableInferenceAdapter:
    def __init__(self, model_version: str = "unconfigured") -> None:
        self.model_version = model_version

    def predict(self, request: InferenceRequest) -> InferenceOutput:
        raise InferenceUnavailable("configured inference backend is unavailable")


def build_inference_adapter(
    runtime_mode: RuntimeMode,
    settings: InferenceSettings,
):
    backend = settings.backend
    if backend is None:
        return DemoInferenceAdapter() if runtime_mode is RuntimeMode.DEMO else UnavailableInferenceAdapter()
    if backend is InferenceBackend.DEMO:
        if runtime_mode is not RuntimeMode.DEMO:
            raise ValueError("demo backend is allowed only in APP_MODE=demo")
        return DemoInferenceAdapter()
    if backend is InferenceBackend.ULTRALYTICS:
        if settings.model_path is None or settings.metadata_path is None:
            return UnavailableInferenceAdapter("ultralytics:unconfigured")
        return UltralyticsInferenceAdapter(
            model_path=settings.model_path,
            metadata_path=settings.metadata_path,
            device=settings.device,
            imgsz=settings.imgsz,
            conf=settings.conf,
        )
    if settings.model_path is None:
        return UnavailableInferenceAdapter("tensorrt:unconfigured")
    return TensorRtInferenceAdapter(settings.model_path)
