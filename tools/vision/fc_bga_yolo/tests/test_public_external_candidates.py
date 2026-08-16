import json
from pathlib import Path
import shutil

from PIL import Image
import pytest

from tools.vision.fc_bga_yolo.model_metadata import sha256_file
from tools.vision.fc_bga_yolo.prepare_public_external_candidates import prepare_candidates
from tools.vision.fc_bga_yolo.public_external_manifest import (
    load_candidate_manifest,
    load_source_registry,
)


SOURCE_ID = "source-001"


def _public_source(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "source"
    paths = (
        root / "train/images/a.rf.111.png",
        root / "val/images/a.rf.222.png",
        root / "test/images/b.rf.333.png",
    )
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (256, 256), "white").save(paths[0])
    shutil.copy2(paths[0], paths[1])
    Image.new("RGB", (256, 256), "black").save(paths[2])
    manifest = root / "source-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "accessed_on": "2026-08-16",
                "format": "yolov8",
                "license": "CC BY 4.0",
                "project": "test-project",
                "purpose": "public_smoke",
                "source_url": "https://example.test/dataset",
                "version": 1,
                "workspace": "test",
                "files": {
                    path.relative_to(root).as_posix(): sha256_file(path)
                    for path in paths
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    registry = tmp_path / "sources.json"
    registry.write_text(
        json.dumps(
            [
                {
                    "source_id": SOURCE_ID,
                    "name": "test source",
                    "url": "https://example.test/dataset",
                    "version": "1",
                    "license_name": "CC BY 4.0",
                    "license_url": "https://creativecommons.org/licenses/by/4.0/legalcode.txt",
                    "license_sha256": "a" * 64,
                    "retrieved_at": "2026-08-16T00:00:00Z",
                    "attribution": "test source, CC BY 4.0",
                }
            ]
        ),
        encoding="utf-8",
    )
    return root, manifest, registry


def test_prepare_candidates_deduplicates_and_marks_review_required(tmp_path: Path) -> None:
    root, manifest, registry = _public_source(tmp_path)

    report = prepare_candidates(
        root,
        manifest,
        registry,
        tmp_path / "review",
        source_id=SOURCE_ID,
    )
    records = load_candidate_manifest(report.manifest, load_source_registry(registry))

    assert report.source_images == 3
    assert report.unique_images == 2
    assert report.exact_duplicates == 1
    assert {record.review_status for record in records} == {"review_required"}
    assert all(record.label_path is None and not record.accepted_classes for record in records)


def test_prepare_candidates_rejects_source_hash_drift(tmp_path: Path) -> None:
    root, manifest, registry = _public_source(tmp_path)
    document = json.loads(manifest.read_text(encoding="utf-8"))
    first = next(iter(document["files"]))
    document["files"][first] = "0" * 64
    manifest.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="SOURCE_HASH_MISMATCH"):
        prepare_candidates(root, manifest, registry, tmp_path / "review", source_id=SOURCE_ID)


def test_prepare_candidates_refuses_changed_output_reuse(tmp_path: Path) -> None:
    root, manifest, registry = _public_source(tmp_path)
    destination = tmp_path / "review"
    prepare_candidates(root, manifest, registry, destination, source_id=SOURCE_ID)
    next((destination / "images").iterdir()).write_bytes(b"changed")

    with pytest.raises(ValueError, match="OUTPUT_CONTENT_CONFLICT"):
        prepare_candidates(root, manifest, registry, destination, source_id=SOURCE_ID)
