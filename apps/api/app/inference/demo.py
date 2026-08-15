from app.inference.base import Detection, InferenceOutput, InferenceRequest


class DemoInferenceAdapter:
    model_version = "yolov8s-aoi-demo"

    def predict(self, request: InferenceRequest) -> InferenceOutput:
        if request.scenario == "DEFECT":
            return InferenceOutput(
                model_version=self.model_version,
                normal_confidence=0.08,
                defect_score=0.92,
                defect_code="BALL_BRIDGE",
                detections=(Detection(32, 24, 42, 36, 0, "BALL_BRIDGE", 0.92),),
                latency_ms=28,
            )
        if request.scenario in {"MISSING_3D", "MISSING_LIGHT"} or not request.input_complete:
            return InferenceOutput(self.model_version, 0.71, 0.18, None, (), 22)
        if request.scenario == "REVIEW":
            return InferenceOutput(self.model_version, 0.72, 0.31, None, (), 25)
        return InferenceOutput(self.model_version, 0.985, 0.015, None, (), 21)
