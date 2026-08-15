from app.inference.base import InferenceOutput, InferenceRequest


class DemoInferenceAdapter:
    def predict(self, request: InferenceRequest) -> InferenceOutput:
        if request.scenario == "DEFECT":
            return InferenceOutput("yolov8s-aoi-demo", 0.08, 0.92, "BALL_BRIDGE", ((32, 24, 42, 36),), 28)
        if request.scenario in {"MISSING_3D", "MISSING_LIGHT"} or not request.input_complete:
            return InferenceOutput("yolov8s-aoi-demo", 0.71, 0.18, None, (), 22)
        if request.scenario == "REVIEW":
            return InferenceOutput("yolov8s-aoi-demo", 0.72, 0.31, "UNKNOWN", ((58, 42, 24, 18),), 25)
        return InferenceOutput("yolov8s-aoi-demo", 0.985, 0.015, None, (), 21)
