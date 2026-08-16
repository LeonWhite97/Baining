from pathlib import Path

import pytest

from tools.vision.fc_bga_yolo.artifact_manifest import (
    capture_artifact_record,
    load_artifact_records,
    verify_artifact_record,
    write_artifact_records,
)


def test_artifact_record_round_trip_detects_tampering(tmp_path: Path) -> None:
    artifact = tmp_path / "yolov8n.pt"
    artifact.write_bytes(b"x" * (1024 * 1024))
    manifest = tmp_path / "artifact-manifest.json"
    record = capture_artifact_record(
        artifact,
        source_url="https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt",
        license_url="https://www.ultralytics.com/license",
        retrieved_at="2026-08-16T12:00:00Z",
    )

    write_artifact_records(manifest, (record,))

    loaded = load_artifact_records(manifest)
    assert loaded == (record,)
    assert verify_artifact_record(artifact, loaded[0]) == record
    artifact.write_bytes(b"y" * (1024 * 1024))
    with pytest.raises(ValueError, match="ARTIFACT_HASH_MISMATCH"):
        verify_artifact_record(artifact, record)


def test_artifact_manifest_rejects_duplicate_names(tmp_path: Path) -> None:
    artifact = tmp_path / "yolov8n.pt"
    artifact.write_bytes(b"x" * (1024 * 1024))
    record = capture_artifact_record(
        artifact,
        source_url="https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt",
        license_url="https://www.ultralytics.com/license",
        retrieved_at="2026-08-16T12:00:00Z",
    )

    with pytest.raises(ValueError, match="ARTIFACT_NAME_DUPLICATE"):
        write_artifact_records(tmp_path / "manifest.json", (record, record))
