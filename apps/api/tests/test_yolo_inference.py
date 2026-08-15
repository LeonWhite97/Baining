from hashlib import sha256
import json
from pathlib import Path

from PIL import Image
import pytest

from app.config import InferenceSettings, RuntimeMode
from app.inference.base import InferenceImage, InferenceRequest, InferenceUnavailable
from app.inference.demo import DemoInferenceAdapter
from app.inference.factory import UnavailableInferenceAdapter, build_inference_adapter
from app.inference.preprocessing import stack_rgb_grayscale
from app.inference.ultralytics import UltralyticsInferenceAdapter


DEFECT_NAMES = (
    "BALL_BRIDGE",
    "MISSING_BALL",
    "EXTRA_BALL",
    "BALL_SIZE_ABNORMAL",
    "BALL_OFFSET",
    "BALL_SHAPE_ABNORMAL",
    "FOREIGN_MATERIAL",
)


def _images(tmp_path: Path) -> tuple[InferenceImage, ...]:
    items: list[InferenceImage] = []
    for light, value in (("R", 20), ("G", 100), ("B", 220), ("RING", 7)):
        path = tmp_path / f"{light}.png"
        Image.new("L", (2, 2), value).save(path)
        items.append(
            InferenceImage(
                light_id=light,
                path=path,
                sha256=sha256(path.read_bytes()).hexdigest(),
                width=2,
                height=2,
                media_type="image/png",
            )
        )
    return tuple(items)


def _model_package(tmp_path: Path) -> tuple[Path, Path]:
    model = tmp_path / "best.pt"
    model.write_bytes(b"model-fixture")
    metadata = tmp_path / "model_metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "dataset_manifest_sha256": "a" * 64,
                "export_settings": {},
                "imgsz": 1280,
                "input_contract": "rgb_grayscale_stack_v1",
                "intended_use": "portfolio_internal_poc",
                "model_sha256": sha256(model.read_bytes()).hexdigest(),
                "model_version": "fc-bga-test-v1",
                "names": list(DEFECT_NAMES),
                "onnx_sha256": None,
                "result_paths": {},
                "runtime_versions": {"python": "test"},
                "task": "detect",
            }
        ),
        encoding="utf-8",
    )
    return model, metadata


class FakeModel:
    def __init__(self) -> None:
        self.calls = 0

    def predict(self, **kwargs: object) -> list[list[tuple[float, ...]]]:
        self.calls += 1
        assert kwargs["imgsz"] == 1280
        assert kwargs["conf"] == 0.25
        return [[(1.0, 2.0, 11.0, 22.0, 0.91, 0.0), (4.0, 5.0, 14.0, 25.0, 0.80, 6.0)]]


def test_runtime_stack_uses_r_g_b_and_requires_ring(tmp_path: Path) -> None:
    images = _images(tmp_path)
    stacked = stack_rgb_grayscale(images)
    assert stacked.getpixel((0, 0)) == (20, 100, 220)
    with pytest.raises(ValueError, match="LIGHT_SET_INVALID"):
        stack_rgb_grayscale(images[:-1])


def test_demo_output_exposes_structured_detection() -> None:
    output = DemoInferenceAdapter().predict(
        InferenceRequest(event_uuid="event-42", scenario="DEFECT", input_complete=True)
    )
    assert output.detections[0].defect_code == "BALL_BRIDGE"
    assert output.detections[0].confidence == output.defect_score
    assert output.boxes == ((32, 24, 42, 36),)


def test_backend_defaults_are_fail_closed_outside_demo() -> None:
    assert isinstance(
        build_inference_adapter(RuntimeMode.DEMO, InferenceSettings.from_values()),
        DemoInferenceAdapter,
    )
    assert isinstance(
        build_inference_adapter(RuntimeMode.SHADOW, InferenceSettings.from_values()),
        UnavailableInferenceAdapter,
    )
    with pytest.raises(ValueError, match="demo backend"):
        build_inference_adapter(
            RuntimeMode.CONTROLLED,
            InferenceSettings.from_values(backend="demo"),
        )


def test_ultralytics_adapter_loads_once_and_keeps_all_detections(tmp_path: Path) -> None:
    model_path, metadata_path = _model_package(tmp_path)
    fake = FakeModel()
    loader_calls: list[Path] = []

    def load(path: Path) -> FakeModel:
        loader_calls.append(path)
        return fake

    adapter = UltralyticsInferenceAdapter(
        model_path=model_path,
        metadata_path=metadata_path,
        device="cpu",
        imgsz=1280,
        conf=0.25,
        model_loader=load,
    )
    assert loader_calls == []
    request = InferenceRequest("event-42", "REVIEW", True, _images(tmp_path))

    first = adapter.predict(request)
    second = adapter.predict(request)

    assert loader_calls == [model_path]
    assert fake.calls == 2
    assert first == second
    assert first.model_version == "fc-bga-test-v1"
    assert first.defect_code == "BALL_BRIDGE"
    assert first.defect_score == 0.91
    assert [(item.class_id, item.defect_code) for item in first.detections] == [
        (0, "BALL_BRIDGE"),
        (6, "FOREIGN_MATERIAL"),
    ]


def test_ultralytics_adapter_rechecks_model_hash(tmp_path: Path) -> None:
    model_path, metadata_path = _model_package(tmp_path)
    adapter = UltralyticsInferenceAdapter(
        model_path=model_path,
        metadata_path=metadata_path,
        device="cpu",
        imgsz=1280,
        conf=0.25,
        model_loader=lambda _: FakeModel(),
    )
    request = InferenceRequest("event-42", "REVIEW", True, _images(tmp_path))
    adapter.predict(request)
    model_path.write_bytes(b"tampered-model")

    with pytest.raises(InferenceUnavailable, match="integrity"):
        adapter.predict(request)
