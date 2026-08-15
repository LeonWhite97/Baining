from dataclasses import replace
from pathlib import Path

import pytest

from tools.vision.fc_bga_yolo.contracts import DEFECT_NAMES, INPUT_CONTRACT
from tools.vision.fc_bga_yolo.model_metadata import (
    ModelMetadata,
    sha256_file,
    validate_model_package,
    write_model_metadata,
)


def _metadata(model: Path) -> ModelMetadata:
    return ModelMetadata(
        model_version="fc-bga-poc-v1",
        task="detect",
        names=DEFECT_NAMES,
        input_contract=INPUT_CONTRACT,
        imgsz=1280,
        dataset_manifest_sha256="a" * 64,
        model_sha256=sha256_file(model),
        onnx_sha256=None,
        runtime_versions={"python": "3.12"},
        export_settings={},
        result_paths={"test": "runs/test/results.csv"},
        intended_use="portfolio_internal_poc",
    )


def test_model_package_round_trip(tmp_path: Path) -> None:
    model = tmp_path / "best.pt"
    model.write_bytes(b"x" * (2 * 1024 * 1024))
    metadata_path = write_model_metadata(tmp_path / "model_metadata.json", _metadata(model))
    loaded = validate_model_package(model, metadata_path, DEFECT_NAMES)
    assert loaded.model_version == "fc-bga-poc-v1"


def test_changed_model_is_rejected(tmp_path: Path) -> None:
    model = tmp_path / "best.pt"
    model.write_bytes(b"x" * (2 * 1024 * 1024))
    metadata_path = write_model_metadata(tmp_path / "metadata.json", _metadata(model))
    model.write_bytes(b"y" + model.read_bytes()[1:])
    with pytest.raises(ValueError, match="MODEL_HASH_MISMATCH"):
        validate_model_package(model, metadata_path, DEFECT_NAMES)


def test_changed_class_order_is_rejected(tmp_path: Path) -> None:
    model = tmp_path / "best.pt"
    model.write_bytes(b"x" * (2 * 1024 * 1024))
    metadata = replace(_metadata(model), names=tuple(reversed(DEFECT_NAMES)))
    metadata_path = write_model_metadata(tmp_path / "metadata.json", metadata)
    with pytest.raises(ValueError, match="MODEL_CLASS_MISMATCH"):
        validate_model_package(model, metadata_path, DEFECT_NAMES)


def test_wrong_input_contract_is_rejected(tmp_path: Path) -> None:
    model = tmp_path / "best.pt"
    model.write_bytes(b"x" * (2 * 1024 * 1024))
    metadata_path = write_model_metadata(
        tmp_path / "metadata.json",
        replace(_metadata(model), input_contract="plain_rgb"),
    )
    with pytest.raises(ValueError, match="MODEL_INPUT_CONTRACT_MISMATCH"):
        validate_model_package(model, metadata_path, DEFECT_NAMES)

