from hashlib import sha256
import json
from pathlib import Path

from PIL import Image
import pytest

from tools.vision.fc_bga_yolo.contracts import DEFECT_NAMES, INPUT_CONTRACT, REQUIRED_LIGHTS
from tools.vision.fc_bga_yolo.validate_yolo_dataset import validate_dataset


@pytest.fixture
def dataset_root(tmp_path: Path) -> Path:
    for split in ("train", "val", "test"):
        (tmp_path / split / "images").mkdir(parents=True)
        (tmp_path / split / "labels").mkdir(parents=True)
    Image.new("RGB", (4, 4), "white").save(tmp_path / "train/images/sample.png")
    (tmp_path / "train/labels/sample.txt").write_text("", encoding="utf-8")
    return tmp_path


def _write_complete_manifest(root: Path, *, sample_id: str = "sample", split: str = "train") -> Path:
    image = root / split / "images" / f"{sample_id}.png"
    label = root / split / "labels" / f"{sample_id}.txt"
    manifest = root / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "sample_id": sample_id,
                "group_id": "lot-1",
                "split": split,
                "input_contract": INPUT_CONTRACT,
                "input_sha256": {light: "a" * 64 for light in REQUIRED_LIGHTS},
                "label_sha256": sha256(label.read_bytes()).hexdigest(),
                "output_image": image.relative_to(root).as_posix(),
                "output_label": label.relative_to(root).as_posix(),
                "output_sha256": sha256(image.read_bytes()).hexdigest(),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


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


def test_manifest_detects_structurally_valid_label_tampering(dataset_root: Path) -> None:
    manifest = _write_complete_manifest(dataset_root)
    (dataset_root / "train/labels/sample.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n",
        encoding="utf-8",
    )

    report = validate_dataset(dataset_root, DEFECT_NAMES, manifest)

    assert "LABEL_HASH_MISMATCH:sample" in report.errors


def test_manifest_rejects_stale_tree_artifacts(dataset_root: Path) -> None:
    manifest = _write_complete_manifest(dataset_root)
    Image.new("RGB", (4, 4), "black").save(dataset_root / "val/images/stale.png")
    (dataset_root / "val/labels/stale.txt").write_text("", encoding="utf-8")

    report = validate_dataset(dataset_root, DEFECT_NAMES, manifest)

    assert "UNMANIFESTED_IMAGE:val/images/stale.png" in report.errors
    assert "UNMANIFESTED_LABEL:val/labels/stale.txt" in report.errors


def test_manifest_rejects_nested_tree_artifacts(dataset_root: Path) -> None:
    manifest = _write_complete_manifest(dataset_root)
    nested_image = dataset_root / "val/images/nested/stale.png"
    nested_label = dataset_root / "val/labels/nested/stale.txt"
    nested_image.parent.mkdir()
    nested_label.parent.mkdir()
    Image.new("RGB", (4, 4), "black").save(nested_image)
    nested_label.write_text("", encoding="utf-8")

    report = validate_dataset(dataset_root, DEFECT_NAMES, manifest)

    assert "UNMANIFESTED_IMAGE:val/images/nested/stale.png" in report.errors
    assert "UNMANIFESTED_LABEL:val/labels/nested/stale.txt" in report.errors


def test_image_directory_rejects_unsupported_files(dataset_root: Path) -> None:
    manifest = _write_complete_manifest(dataset_root)
    (dataset_root / "val/images/stale.bmp").write_bytes(b"not-a-training-image")

    report = validate_dataset(dataset_root, DEFECT_NAMES, manifest)

    assert "UNSUPPORTED_IMAGE_FILE:val/images/stale.bmp" in report.errors


def test_label_directory_rejects_unsupported_files(dataset_root: Path) -> None:
    manifest = _write_complete_manifest(dataset_root)
    (dataset_root / "val/labels/stale.csv").write_text("ignored", encoding="utf-8")

    report = validate_dataset(dataset_root, DEFECT_NAMES, manifest)

    assert "UNSUPPORTED_LABEL_FILE:val/labels/stale.csv" in report.errors
