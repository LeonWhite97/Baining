import hashlib
import json
from pathlib import Path

from PIL import Image
import pytest

from tools.vision.fc_bga_yolo.convert_dataset import convert_manifest, parse_source_manifest
from tools.vision.fc_bga_yolo.preprocessing import stack_rgb_grayscale


def _write_manifest(tmp_path: Path, *, image_size: tuple[int, int] = (2, 2)) -> Path:
    images: dict[str, str] = {}
    for light, value in (("R", 20), ("G", 100), ("B", 220), ("RING", 7)):
        path = tmp_path / f"{light}.png"
        Image.new("L", image_size, value).save(path)
        images[light] = path.name
    (tmp_path / "sample-1.txt").write_text("", encoding="utf-8")
    manifest = tmp_path / "source.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "sample_id": "sample-1",
                "group_id": "lot-1",
                "split": "train",
                "images": images,
                "label": "sample-1.txt",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def test_stack_uses_r_g_b_grayscale_channel_order(tmp_path: Path) -> None:
    paths = []
    for name, value in (("R", 20), ("G", 100), ("B", 220)):
        path = tmp_path / f"{name}.png"
        Image.new("L", (2, 2), value).save(path)
        paths.append(path)
    stacked = stack_rgb_grayscale(*paths)
    assert stacked.mode == "RGB"
    assert stacked.getpixel((0, 0)) == (20, 100, 220)


def test_convert_requires_ring_but_does_not_use_it_as_a_channel(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path)
    ring_path = tmp_path / "RING.png"
    output_root = tmp_path / "dataset"

    report = convert_manifest(manifest, output_root)

    output_image = output_root / "train/images/sample-1.png"
    assert Image.open(output_image).getpixel((0, 0)) == (20, 100, 220)
    assert (output_root / "train/labels/sample-1.txt").read_text(encoding="utf-8") == ""
    provenance = json.loads(report.output_manifest.read_text(encoding="utf-8").splitlines()[0])
    assert provenance["input_contract"] == "rgb_grayscale_stack_v1"
    assert provenance["input_sha256"]["RING"] == hashlib.sha256(ring_path.read_bytes()).hexdigest()


def test_missing_ring_is_rejected_without_output(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path)
    record = json.loads(manifest.read_text(encoding="utf-8"))
    del record["images"]["RING"]
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="LIGHT_SET_INVALID"):
        convert_manifest(manifest, tmp_path / "dataset")
    assert not (tmp_path / "dataset/train/images/sample-1.png").exists()


def test_unequal_dimensions_are_rejected(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path)
    Image.new("L", (3, 2), 7).save(tmp_path / "RING.png")
    with pytest.raises(ValueError, match="IMAGE_SIZE_MISMATCH"):
        convert_manifest(manifest, tmp_path / "dataset")


def test_duplicate_sample_ids_are_rejected(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path)
    manifest.write_text(manifest.read_text(encoding="utf-8") * 2, encoding="utf-8")
    with pytest.raises(ValueError, match="DUPLICATE_SAMPLE_ID"):
        parse_source_manifest(manifest)


def test_paths_outside_manifest_root_are_rejected(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path)
    record = json.loads(manifest.read_text(encoding="utf-8"))
    record["label"] = "../outside.txt"
    (tmp_path.parent / "outside.txt").write_text("", encoding="utf-8")
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="PATH_OUTSIDE_ROOT"):
        parse_source_manifest(manifest)


def test_invalid_yolo_label_is_rejected(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path)
    (tmp_path / "sample-1.txt").write_text("0 0.5 0.5 0.2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="LABEL_FORMAT_INVALID"):
        convert_manifest(manifest, tmp_path / "dataset")


def test_late_invalid_record_leaves_existing_output_untouched(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path)
    first = json.loads(manifest.read_text(encoding="utf-8"))
    second_images: dict[str, str] = {}
    for light, value in (("R", 30), ("G", 110), ("B", 210), ("RING", 9)):
        path = tmp_path / f"{light}2.png"
        size = (3, 2) if light == "RING" else (2, 2)
        Image.new("L", size, value).save(path)
        second_images[light] = path.name
    (tmp_path / "sample-2.txt").write_text("", encoding="utf-8")
    second = {
        "sample_id": "sample-2",
        "group_id": "lot-2",
        "split": "val",
        "images": second_images,
        "label": "sample-2.txt",
    }
    manifest.write_text(
        json.dumps(first) + "\n" + json.dumps(second) + "\n",
        encoding="utf-8",
    )
    output_root = tmp_path / "dataset"
    (output_root / "train/images").mkdir(parents=True)
    old_image = output_root / "train/images/old.png"
    Image.new("RGB", (1, 1), "red").save(old_image)
    old_manifest = output_root / "manifest.jsonl"
    old_manifest.write_text("old-state\n", encoding="utf-8")

    with pytest.raises(ValueError, match="IMAGE_SIZE_MISMATCH: sample-2"):
        convert_manifest(manifest, output_root)

    assert old_image.is_file()
    assert old_manifest.read_text(encoding="utf-8") == "old-state\n"
    assert not (output_root / "train/images/sample-1.png").exists()


def test_successful_conversion_replaces_stale_dataset_and_preserves_scaffold(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path)
    output_root = tmp_path / "dataset"
    (output_root / "train/images").mkdir(parents=True)
    stale = output_root / "train/images/stale.png"
    Image.new("RGB", (1, 1), "red").save(stale)
    (output_root / "README.md").write_text("keep me\n", encoding="utf-8")

    convert_manifest(manifest, output_root)

    assert not stale.exists()
    assert (output_root / "train/images/sample-1.png").is_file()
    assert (output_root / "README.md").read_text(encoding="utf-8") == "keep me\n"
