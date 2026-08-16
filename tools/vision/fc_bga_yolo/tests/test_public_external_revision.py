from dataclasses import replace
import json
from pathlib import Path

from PIL import Image
import pytest

from tools.vision.fc_bga_yolo.contracts import DEFECT_NAMES
from tools.vision.fc_bga_yolo.model_metadata import sha256_file
from tools.vision.fc_bga_yolo.public_external_manifest import CandidateRecord
from tools.vision.fc_bga_yolo.public_external_revision import (
    assign_group_stratified_v1,
    evaluate_revision_gate,
    publish_revision,
)


def _accepted_records(
    tmp_path: Path,
    count: int,
    classes: tuple[int, ...],
) -> tuple[CandidateRecord, ...]:
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir(exist_ok=True)
    labels.mkdir(exist_ok=True)
    result = []
    for index in range(count):
        sample_id = f"sample-{index:03d}"
        image = images / f"{sample_id}.png"
        label = labels / f"{sample_id}.txt"
        Image.new("RGB", (256, 256), (index % 256, 0, 0)).save(image)
        label.write_text(
            "".join(f"{class_id} 0.5 0.5 0.25 0.25\n" for class_id in classes),
            encoding="ascii",
        )
        result.append(
            CandidateRecord(
                sample_id=sample_id,
                source_group_id=f"group-{index:03d}",
                source_id="source-001",
                original_filename=image.name,
                image_path=image.relative_to(tmp_path).as_posix(),
                image_sha256=sha256_file(image),
                label_path=label.relative_to(tmp_path).as_posix(),
                review_status="accepted",
                annotation_status="provisional_human_reviewed_poc",
                accepted_classes=tuple(DEFECT_NAMES[class_id] for class_id in classes),
                quarantine_reason=None,
            )
        )
    return tuple(result)


def test_b0_gate_requires_twenty_images_and_two_classes(tmp_path: Path) -> None:
    blocked = _accepted_records(tmp_path, 19, (0, 1))
    ready = _accepted_records(tmp_path, 20, (0, 1))

    assert evaluate_revision_gate(blocked, "B0", manifest_root=tmp_path).status == "blocked_data"
    assert evaluate_revision_gate(ready, "B0", manifest_root=tmp_path).status == "ready"


def test_group_split_is_deterministic_nonempty_and_leakage_free(tmp_path: Path) -> None:
    records = list(_accepted_records(tmp_path, 20, (0, 1)))
    records[1] = replace(records[1], source_group_id=records[0].source_group_id)

    first = assign_group_stratified_v1(tuple(records), seed=42)
    second = assign_group_stratified_v1(tuple(reversed(records)), seed=42)

    assert first == second
    assert set(first.values()) == {"train", "val", "test"}
    assert first[records[0].sample_id] == first[records[1].sample_id]


def test_v02_rebuilds_all_splits_and_preserves_v01(tmp_path: Path) -> None:
    versions = tmp_path / "versions"
    v01 = publish_revision(
        _accepted_records(tmp_path, 20, (0, 1)),
        tmp_path,
        versions,
        version="public-external-v0.1",
        stage="B0",
    )
    old_assignments = v01.assignments.read_bytes()
    v02 = publish_revision(
        _accepted_records(tmp_path, 100, (0, 1, 2)),
        tmp_path,
        versions,
        version="public-external-v0.2",
        stage="B1",
    )
    assignments = [
        json.loads(line)
        for line in v02.assignments.read_text(encoding="utf-8").splitlines()
    ]

    assert len(assignments) == 100
    assert v01.assignments.read_bytes() == old_assignments
    assert v01.manifest_sha256 != v02.manifest_sha256


def test_revision_refuses_changed_content_at_existing_version(tmp_path: Path) -> None:
    versions = tmp_path / "versions"
    publish_revision(
        _accepted_records(tmp_path, 20, (0, 1)),
        tmp_path,
        versions,
        version="public-external-v0.1",
        stage="B0",
    )

    with pytest.raises(ValueError, match="REVISION_IMMUTABLE"):
        publish_revision(
            _accepted_records(tmp_path, 21, (0, 1)),
            tmp_path,
            versions,
            version="public-external-v0.1",
            stage="B0",
        )
