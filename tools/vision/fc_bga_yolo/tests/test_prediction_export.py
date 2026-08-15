from hashlib import sha256
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

from PIL import Image
import pytest

from tools.vision.fc_bga_yolo.contracts import DEFECT_NAMES, INPUT_CONTRACT
from tools.vision.fc_bga_yolo.export_model import export_model
from tools.vision.fc_bga_yolo.model_metadata import (
    ModelMetadata,
    load_model_metadata,
    write_model_metadata,
)
from tools.vision.fc_bga_yolo.predict import predict, resolve_prediction_inputs


class _Rows:
    def __init__(self, rows: list[list[float]]) -> None:
        self._rows = rows

    def tolist(self) -> list[list[float]]:
        return self._rows


class _Result:
    def __init__(self, rows: list[list[float]]) -> None:
        self.boxes = SimpleNamespace(data=_Rows(rows))


class FakeModel:
    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self.names = {index: name for index, name in enumerate(DEFECT_NAMES)}
        self.export_kwargs: dict[str, object] | None = None

    def predict(self, **_: object) -> list[_Result]:
        return [_Result([[1.0, 1.0, 3.0, 3.0, 0.9, 0.0]])]

    def export(self, **kwargs: object) -> str:
        self.export_kwargs = kwargs
        output = self.model_path.with_suffix(".onnx")
        output.write_bytes(b"o" * 2048)
        return str(output)


def _model_package(tmp_path: Path) -> tuple[Path, Path]:
    model = tmp_path / "best.pt"
    model.write_bytes(b"formal-model")
    metadata = ModelMetadata(
        model_version="fc-bga-test-v1",
        task="detect",
        names=DEFECT_NAMES,
        input_contract=INPUT_CONTRACT,
        imgsz=1280,
        dataset_manifest_sha256="a" * 64,
        model_sha256=sha256(model.read_bytes()).hexdigest(),
        onnx_sha256=None,
        runtime_versions={"python": "test"},
        export_settings={},
        result_paths={},
        intended_use="portfolio_internal_poc",
    )
    return model, write_model_metadata(tmp_path / "model_metadata.json", metadata)


def test_prediction_writes_annotations_summary_and_configuration_hash(tmp_path: Path) -> None:
    model_path, metadata_path = _model_package(tmp_path)
    source = tmp_path / "sample.png"
    Image.new("RGB", (4, 4), "white").save(source)
    output = tmp_path / "predictions.jsonl"
    fake = FakeModel(model_path)

    returned = predict(
        model_path,
        metadata_path,
        source,
        output,
        conf=0.25,
        device="cpu",
        model_loader=lambda _: fake,
    )

    assert returned == output
    record = json.loads(output.read_text(encoding="utf-8"))
    assert len(record["configuration_sha256"]) == 64
    assert Path(record["annotated_image"]).is_file()
    summary = json.loads((tmp_path / "predictions.summary.json").read_text(encoding="utf-8"))
    assert summary["configuration_sha256"] == record["configuration_sha256"]
    assert summary["classes"]["BALL_BRIDGE"] == {
        "box_count": 1,
        "confidence_max": 0.9,
        "confidence_mean": 0.9,
        "confidence_min": 0.9,
        "image_count": 1,
    }


def test_prediction_rejects_actual_model_class_mismatch(tmp_path: Path) -> None:
    model_path, metadata_path = _model_package(tmp_path)
    source = tmp_path / "sample.png"
    Image.new("RGB", (4, 4), "white").save(source)
    fake = FakeModel(model_path)
    fake.names = {0: "OK", 1: "NG"}

    with pytest.raises(ValueError, match="MODEL_CLASS_MISMATCH"):
        predict(
            model_path,
            metadata_path,
            source,
            tmp_path / "predictions.jsonl",
            conf=0.25,
            device="cpu",
            model_loader=lambda _: fake,
        )


def test_prediction_manifest_rejects_path_outside_manifest_root(tmp_path: Path) -> None:
    manifest_root = tmp_path / "dataset"
    manifest_root.mkdir()
    outside = tmp_path / "outside.png"
    Image.new("RGB", (4, 4), "white").save(outside)
    manifest = manifest_root / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"output_image": "../outside.png"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="PREDICTION_PATH_OUTSIDE_ROOT"):
        resolve_prediction_inputs(manifest)


def test_onnx_export_records_explicit_reproducibility_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path, metadata_path = _model_package(tmp_path)
    fake = FakeModel(model_path)
    fake_torch = SimpleNamespace(
        __version__="test-torch",
        version=SimpleNamespace(cuda=None),
        cuda=SimpleNamespace(is_available=lambda: False),
    )
    fake_ultralytics = ModuleType("ultralytics")
    fake_ultralytics.YOLO = lambda _: fake
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "ultralytics", fake_ultralytics)
    monkeypatch.setattr(
        "tools.vision.fc_bga_yolo.export_model.importlib.metadata.version",
        lambda _: "test-ultralytics",
    )

    exported = export_model(
        model_path,
        metadata_path,
        format_name="onnx",
        imgsz=1024,
        device="cpu",
        opset=17,
        dynamic=True,
        simplify=True,
    )

    assert exported.is_file()
    assert fake.export_kwargs == {
        "format": "onnx",
        "imgsz": 1024,
        "device": "cpu",
        "opset": 17,
        "dynamic": True,
        "simplify": True,
    }
    metadata = load_model_metadata(metadata_path)
    assert metadata.export_settings["opset"] == 17
    assert metadata.export_settings["dynamic"] is True
    assert metadata.export_settings["simplify"] is True
    assert metadata.export_settings["output_sha256"] == sha256(exported.read_bytes()).hexdigest()
