import json
from pathlib import Path

from PIL import Image
import pytest

from tools.vision.fc_bga_yolo.convert_dataset import convert_manifest
from tools.vision.fc_bga_yolo.contracts import DEFECT_NAMES
from tools.vision.fc_bga_yolo.mock_capture_dataset import create_mock_capture_dataset
from tools.vision.fc_bga_yolo.validate_yolo_dataset import validate_dataset


def _write_input_images(root: Path, count: int) -> None:
    root.mkdir()
    for index in range(1, count + 1):
        Image.new("RGB", (4, 3), (index, index + 10, index + 20)).save(
            root / f"capture-{index}.jpg"
        )


def test_mock_capture_dataset_writes_four_light_manifest_and_empty_labels(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_input_images(source, 5)
    output = tmp_path / "mock_capture"

    report = create_mock_capture_dataset(
        source,
        output,
        prefix="MOCK",
        group_id="mock_lot_001",
        train_ratio=0.6,
        val_ratio=0.2,
    )

    assert report.samples == 5
    assert dict(report.split_counts) == {"train": 3, "val": 1, "test": 1}
    records = [
        json.loads(line)
        for line in (output / "source.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [record["sample_id"] for record in records] == [
        "MOCK0001",
        "MOCK0002",
        "MOCK0003",
        "MOCK0004",
        "MOCK0005",
    ]
    assert records[0] == {
        "sample_id": "MOCK0001",
        "group_id": "mock_lot_001",
        "split": "train",
        "images": {
            "R": "raw/MOCK0001_R.png",
            "G": "raw/MOCK0001_G.png",
            "B": "raw/MOCK0001_B.png",
            "RING": "raw/MOCK0001_RING.png",
        },
        "label": "annotations/MOCK0001.txt",
    }
    for light in ("R", "G", "B", "RING"):
        assert (output / f"raw/MOCK0001_{light}.png").is_file()
    assert (output / "annotations/MOCK0001.txt").read_text(encoding="utf-8") == ""


def test_mock_capture_output_can_be_converted_to_formal_dataset(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_input_images(source, 2)
    output = tmp_path / "mock_capture"

    create_mock_capture_dataset(source, output)
    conversion = convert_manifest(output / "source.jsonl", tmp_path / "formal_dataset")

    assert conversion.samples == 2
    assert (tmp_path / "formal_dataset/train/images/MOCK0001.png").is_file()
    assert (tmp_path / "formal_dataset/train/labels/MOCK0001.txt").read_text(
        encoding="utf-8"
    ) == ""


def test_mock_capture_default_groups_do_not_leak_across_splits(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_input_images(source, 5)
    output = tmp_path / "mock_capture"
    formal = tmp_path / "formal_dataset"

    create_mock_capture_dataset(source, output, train_ratio=0.6, val_ratio=0.2)
    conversion = convert_manifest(output / "source.jsonl", formal)

    report = validate_dataset(formal, DEFECT_NAMES, conversion.output_manifest)
    assert report.errors == ()


def test_mock_capture_rejects_empty_input_directory(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()

    with pytest.raises(ValueError, match="SOURCE_IMAGES_EMPTY"):
        create_mock_capture_dataset(source, tmp_path / "mock_capture")


def test_mock_capture_rejects_nonempty_output_without_force(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_input_images(source, 1)
    output = tmp_path / "mock_capture"
    output.mkdir()
    (output / "existing.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(ValueError, match="OUTPUT_NOT_EMPTY"):
        create_mock_capture_dataset(source, output)
    assert (output / "existing.txt").read_text(encoding="utf-8") == "keep"
