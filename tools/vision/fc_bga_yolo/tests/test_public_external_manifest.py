from dataclasses import replace
from pathlib import Path

from PIL import Image

from tools.vision.fc_bga_yolo.model_metadata import sha256_file
from tools.vision.fc_bga_yolo.public_external_manifest import (
    CandidateRecord,
    SourceRecord,
    assess_license_state,
    audit_candidates,
)


def _accepted_candidate(tmp_path: Path) -> tuple[SourceRecord, CandidateRecord]:
    image = tmp_path / "images" / "sample-001.png"
    label = tmp_path / "labels" / "sample-001.txt"
    image.parent.mkdir()
    label.parent.mkdir()
    Image.new("RGB", (256, 256), "white").save(image)
    label.write_text("0 0.5 0.5 0.25 0.25\n", encoding="ascii")
    source = SourceRecord(
        source_id="source-001",
        name="test source",
        url="https://example.test/dataset",
        version="1",
        license_name="CC BY 4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/legalcode.txt",
        license_sha256="a" * 64,
        retrieved_at="2026-08-16T00:00:00Z",
        attribution="test source, CC BY 4.0",
    )
    record = CandidateRecord(
        sample_id="sample-001",
        source_group_id="group-001",
        source_id=source.source_id,
        original_filename=image.name,
        image_path="images/sample-001.png",
        image_sha256=sha256_file(image),
        label_path="labels/sample-001.txt",
        review_status="accepted",
        annotation_status="provisional_human_reviewed_poc",
        accepted_classes=("BALL_BRIDGE",),
        quarantine_reason=None,
    )
    return source, record


def test_accepted_candidate_passes_strict_audit(tmp_path: Path) -> None:
    source, record = _accepted_candidate(tmp_path)

    audit = audit_candidates((record,), sources=(source,), manifest_root=tmp_path)

    assert audit.errors == ()
    assert audit.accepted_images == 1
    assert audit.represented_classes == ("BALL_BRIDGE",)
    assert audit.class_boxes == {"BALL_BRIDGE": 1}


def test_candidate_audit_rejects_path_escape_and_unreviewed_acceptance(tmp_path: Path) -> None:
    source, record = _accepted_candidate(tmp_path)
    escaped = replace(record, image_path="../escape.png")
    unreviewed = replace(record, annotation_status=None)

    escaped_audit = audit_candidates((escaped,), sources=(source,), manifest_root=tmp_path)
    unreviewed_audit = audit_candidates((unreviewed,), sources=(source,), manifest_root=tmp_path)

    assert any("PATH_ESCAPE" in error for error in escaped_audit.errors)
    assert any("ANNOTATION_STATUS_REQUIRED" in error for error in unreviewed_audit.errors)


def test_license_change_quarantines_without_deleting_cache(tmp_path: Path) -> None:
    cached = tmp_path / "image.png"
    Image.new("RGB", (256, 256), "white").save(cached)

    state = assess_license_state(recorded_sha="a" * 64, current_sha="b" * 64)

    assert state == "quarantined_license_change"
    assert cached.is_file()
