from pathlib import Path

from PIL import Image
import pytest

from tools.vision.fc_bga_yolo.deduplicate_yolo_dataset import (
    apply_duplicate_report,
    audit_duplicates,
)


def _duplicate_dataset(tmp_path: Path, labels: tuple[str, str]) -> Path:
    for split in ("train", "test"):
        (tmp_path / split / "images").mkdir(parents=True)
        (tmp_path / split / "labels").mkdir(parents=True)
    image = Image.new("RGB", (4, 4), "white")
    image.save(tmp_path / "train/images/a.png")
    image.save(tmp_path / "test/images/b.png")
    (tmp_path / "train/labels/a.txt").write_text(labels[0], encoding="utf-8")
    (tmp_path / "test/labels/b.txt").write_text(labels[1], encoding="utf-8")
    return tmp_path


def test_audit_only_never_deletes_files(tmp_path: Path) -> None:
    root = _duplicate_dataset(tmp_path, ("0 0.5 0.5 0.2 0.2\n", "0   0.5 0.5 0.2 0.2\n"))
    report = audit_duplicates(root)
    assert report.redundant_images == 1
    assert len(list(root.rglob("*.png"))) == 2
    assert report.groups[0].keep_image.parts[-3] == "test"


def test_apply_removes_only_redundant_matching_pair(tmp_path: Path) -> None:
    root = _duplicate_dataset(tmp_path, ("0 0.5 0.5 0.2 0.2\n", "0 0.5 0.5 0.2 0.2\n"))
    applied = apply_duplicate_report(audit_duplicates(root))
    assert applied.removed_images == 1
    assert not (root / "train/images/a.png").exists()
    assert not (root / "train/labels/a.txt").exists()
    assert (root / "test/images/b.png").exists()


def test_same_image_with_different_labels_is_a_conflict(tmp_path: Path) -> None:
    root = _duplicate_dataset(tmp_path, ("0 0.5 0.5 0.2 0.2\n", "1 0.5 0.5 0.2 0.2\n"))
    report = audit_duplicates(root)
    assert report.conflicts == 1
    with pytest.raises(ValueError, match="LABEL_CONFLICT"):
        apply_duplicate_report(report)
