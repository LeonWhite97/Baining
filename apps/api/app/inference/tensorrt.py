from pathlib import Path

from app.inference.base import InferenceOutput, InferenceRequest, InferenceUnavailable


class TensorRtInferenceAdapter:
    def __init__(self, engine_path: Path) -> None:
        self.engine_path = engine_path
        self.model_version = f"tensorrt:{engine_path.stem}"

    def predict(self, request: InferenceRequest) -> InferenceOutput:
        if not self.engine_path.is_file():
            raise InferenceUnavailable(f"TensorRT engine unavailable: {self.engine_path}")
        raise InferenceUnavailable("TensorRT runtime is not enabled in the GPU-free build")
