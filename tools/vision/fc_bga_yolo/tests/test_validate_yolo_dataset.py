import json
from pathlib import Path

from PIL import Image
import pytest

from tools.vision.fc_bga_yolo.contracts import DEFECT_NAMES
from tools.vision.fc_bga_yolo.validate_yolo_dataset import validate_dataset


@pytest.fixture
def dataset_root(tmp_path: Path) -> Path:
    for split in ("train", "val", "test"):
        (tmp_path / split / "images").mkdir(parents=True)
        (tmp_path / split / "labels").mkdir(parents=True)
    Image.new("RGB", (4, 4), "white").save(tmp_path / "train/images/sample.png")
    (tmp_path / "train/labels/sample.txt").write_text("", encoding="utf-8")
    return tmp_path


def test_empty_label_is_a_valid_normal_sample(dataset_root: Path) -> None:
    report = validate_dataset(dataset_root, DEFECT_NAMES, None)
    assert report.errors == ()
    assert report.empty_labels == 1
    assert report.images == 1


@pytest.mark.parametrize(
    "line",
    ["7 0.5 0.5 0.2 0.2", "0 nan 0.5 0.2 0.2", "0 0.5 0.5 0 0.2"],
)
def test_invalid_label_values_fail(dataset_root: Path, line: str) -> None:
    (dataset_root / "train/labels/sample.txt").write_text(line + "\n", encoding="utf-8")
    assert validate_dataset(dataset_root, DEFECT_NAMES, None).errors


def test_group_leakage_is_reported(dataset_root: Path) -> None:
    Image.new("RGB", (4, 4), "black").save(dataset_root / "test/images/second.png")
    (dataset_root / "test/labels/second.txt").write_text("", encoding="utf-8")
    manifest = dataset_root / "manifest.jsonl"
    manifest.write_text(
        "\n".join(
            [
                json.dumps({"sample_id": "sample", "group_id": "lot-1", "split": "train"}),
                json.dumps({"sample_id": "second", "group_id": "lot-1", "split": "test"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    report = validate_dataset(dataset_root, DEFECT_NAMES, manifest)
    assert any("GROUP_LEAKAGE" in error for error in report.errors)


def test_missing_label_pair_is_reported(dataset_root: Path) -> None:
    Image.new("RGB", (4, 4), "black").save(dataset_root / "val/images/unpaired.png")
    report = validate_dataset(dataset_root, DEFECT_NAMES, None)
    assert any("MISSING_LABEL" in error for error in report.errors)
