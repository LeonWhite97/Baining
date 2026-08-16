from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
import re
from typing import Literal, Mapping
from urllib.parse import urlparse

from PIL import Image, UnidentifiedImageError

try:
    from .contracts import DEFECT_NAMES
    from .model_metadata import sha256_file
    from .validate_yolo_dataset import _validate_label
except ImportError:
    from contracts import DEFECT_NAMES
    from model_metadata import sha256_file
    from validate_yolo_dataset import _validate_label


ReviewStatus = Literal["review_required", "accepted", "quarantined"]
AnnotationStatus = Literal["provisional_human_reviewed_poc"]
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class SourceRecord:
    source_id: str
    name: str
    url: str
    version: str
    license_name: str
    license_url: str
    license_sha256: str
    retrieved_at: str
    attribution: str


@dataclass(frozen=True, slots=True)
class CandidateRecord:
    sample_id: str
    source_group_id: str
    source_id: str
    original_filename: str
    image_path: str
    image_sha256: str
    label_path: str | None
    review_status: ReviewStatus
    annotation_status: AnnotationStatus | None
    accepted_classes: tuple[str, ...]
    quarantine_reason: str | None


@dataclass(frozen=True, slots=True)
class CandidateAudit:
    total_images: int
    accepted_images: int
    represented_classes: tuple[str, ...]
    class_boxes: Mapping[str, int]
    quarantine_counts: Mapping[str, int]
    errors: tuple[str, ...]


def _validate_https(value: str, error: str) -> None:
    parsed = urlparse(value)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError(error)


def _validate_source(source: SourceRecord) -> SourceRecord:
    text_fields = (
        source.source_id,
        source.name,
        source.version,
        source.license_name,
        source.attribution,
    )
    if any(not value.strip() for value in text_fields):
        raise ValueError("SOURCE_FIELD_INVALID")
    _validate_https(source.url, "SOURCE_URL_INVALID")
    _validate_https(source.license_url, "LICENSE_URL_INVALID")
    if not _SHA256.fullmatch(source.license_sha256):
        raise ValueError("LICENSE_SNAPSHOT_REQUIRED")
    try:
        retrieved = datetime.fromisoformat(source.retrieved_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("SOURCE_RETRIEVED_AT_INVALID") from exc
    if not source.retrieved_at.endswith("Z") or retrieved.utcoffset() is None:
        raise ValueError("SOURCE_RETRIEVED_AT_INVALID")
    return source


def load_source_registry(path: Path) -> tuple[SourceRecord, ...]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError) as exc:
        raise ValueError("SOURCE_REGISTRY_INVALID") from exc
    if not isinstance(document, list):
        raise ValueError("SOURCE_REGISTRY_INVALID")
    expected = set(SourceRecord.__dataclass_fields__)
    records: list[SourceRecord] = []
    for item in document:
        if not isinstance(item, dict) or set(item) != expected or any(not isinstance(value, str) for value in item.values()):
            raise ValueError("SOURCE_REGISTRY_INVALID")
        records.append(_validate_source(SourceRecord(**item)))
    if len({record.source_id for record in records}) != len(records):
        raise ValueError("SOURCE_ID_DUPLICATE")
    return tuple(records)


def _candidate_from_json(item: object) -> CandidateRecord:
    expected = set(CandidateRecord.__dataclass_fields__)
    if not isinstance(item, dict) or set(item) != expected:
        raise ValueError("CANDIDATE_RECORD_INVALID")
    accepted = item.get("accepted_classes")
    if not isinstance(accepted, list) or any(not isinstance(value, str) for value in accepted):
        raise ValueError("CANDIDATE_RECORD_INVALID")
    scalar_fields = (
        "sample_id",
        "source_group_id",
        "source_id",
        "original_filename",
        "image_path",
        "image_sha256",
        "review_status",
    )
    if any(not isinstance(item.get(field), str) for field in scalar_fields):
        raise ValueError("CANDIDATE_RECORD_INVALID")
    for field in ("label_path", "annotation_status", "quarantine_reason"):
        if item.get(field) is not None and not isinstance(item.get(field), str):
            raise ValueError("CANDIDATE_RECORD_INVALID")
    return CandidateRecord(
        sample_id=item["sample_id"],
        source_group_id=item["source_group_id"],
        source_id=item["source_id"],
        original_filename=item["original_filename"],
        image_path=item["image_path"],
        image_sha256=item["image_sha256"],
        label_path=item["label_path"],
        review_status=item["review_status"],
        annotation_status=item["annotation_status"],
        accepted_classes=tuple(accepted),
        quarantine_reason=item["quarantine_reason"],
    )


def load_candidate_manifest(
    path: Path,
    sources: tuple[SourceRecord, ...],
) -> tuple[CandidateRecord, ...]:
    source_ids = {source.source_id for source in sources}
    records: list[CandidateRecord] = []
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError as exc:
        raise ValueError("CANDIDATE_MANIFEST_INVALID") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = _candidate_from_json(json.loads(line))
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"CANDIDATE_MANIFEST_INVALID:{line_number}") from exc
        if record.source_id not in source_ids:
            raise ValueError(f"SOURCE_UNREGISTERED:{record.source_id}")
        records.append(record)
    return tuple(records)


def _resolve_inside(root: Path, value: str, sample_id: str) -> tuple[Path | None, str | None]:
    path = (root / value).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        return None, f"PATH_ESCAPE:{sample_id}:{value}"
    return path, None


def audit_candidates(
    records: tuple[CandidateRecord, ...],
    *,
    sources: tuple[SourceRecord, ...],
    manifest_root: Path,
) -> CandidateAudit:
    root = manifest_root.resolve()
    source_ids: set[str] = set()
    errors: list[str] = []
    for source in sources:
        try:
            _validate_source(source)
        except ValueError as exc:
            errors.append(f"{exc}:{source.source_id}")
        source_ids.add(source.source_id)
    if len(source_ids) != len(sources):
        errors.append("SOURCE_ID_DUPLICATE")
    sample_ids: set[str] = set()
    image_hashes: set[str] = set()
    class_boxes: Counter[str] = Counter()
    quarantine_counts: Counter[str] = Counter()
    accepted_images = 0
    for record in records:
        prefix = record.sample_id or "<empty>"
        if not record.sample_id or record.sample_id in sample_ids:
            errors.append(f"SAMPLE_ID_INVALID_OR_DUPLICATE:{prefix}")
        sample_ids.add(record.sample_id)
        if not record.source_group_id:
            errors.append(f"SOURCE_GROUP_REQUIRED:{prefix}")
        if record.source_id not in source_ids:
            errors.append(f"SOURCE_UNREGISTERED:{prefix}")
        if record.review_status not in {"review_required", "accepted", "quarantined"}:
            errors.append(f"REVIEW_STATUS_INVALID:{prefix}")
        if not _SHA256.fullmatch(record.image_sha256):
            errors.append(f"IMAGE_SHA256_INVALID:{prefix}")
        elif record.image_sha256 in image_hashes:
            errors.append(f"EXACT_DUPLICATE_UNRESOLVED:{prefix}")
        image_hashes.add(record.image_sha256)
        image, path_error = _resolve_inside(root, record.image_path, prefix)
        if path_error:
            errors.append(path_error)
            continue
        assert image is not None
        if not image.is_file():
            errors.append(f"IMAGE_UNAVAILABLE:{prefix}")
        else:
            if sha256_file(image) != record.image_sha256:
                errors.append(f"IMAGE_HASH_MISMATCH:{prefix}")
            try:
                with Image.open(image) as opened:
                    opened.verify()
                with Image.open(image) as opened:
                    width, height = opened.size
                if width < 256 or height < 256:
                    errors.append(f"IMAGE_TOO_SMALL:{prefix}")
            except (OSError, UnidentifiedImageError):
                errors.append(f"IMAGE_DECODE_FAILED:{prefix}")
        if record.review_status == "accepted":
            accepted_images += 1
            if record.annotation_status != "provisional_human_reviewed_poc":
                errors.append(f"ANNOTATION_STATUS_REQUIRED:{prefix}")
            if record.quarantine_reason is not None:
                errors.append(f"ACCEPTED_QUARANTINE_REASON_FORBIDDEN:{prefix}")
            if record.label_path is None:
                errors.append(f"ACCEPTED_LABEL_REQUIRED:{prefix}")
                continue
            label, label_path_error = _resolve_inside(root, record.label_path, prefix)
            if label_path_error:
                errors.append(label_path_error)
                continue
            assert label is not None
            boxes, label_errors, counts = _validate_label(label, DEFECT_NAMES)
            errors.extend(f"{error}:{prefix}" for error in label_errors)
            if boxes != sum(counts.values()):
                errors.append(f"LABEL_COUNT_INVALID:{prefix}")
            found_classes = tuple(name for name in DEFECT_NAMES if counts[name])
            if found_classes != record.accepted_classes:
                errors.append(f"ACCEPTED_CLASS_MISMATCH:{prefix}")
            class_boxes.update(counts)
        elif record.review_status == "review_required":
            if record.label_path is not None or record.annotation_status is not None or record.accepted_classes:
                errors.append(f"REVIEW_REQUIRED_STATE_INVALID:{prefix}")
            if record.quarantine_reason is not None:
                errors.append(f"REVIEW_REQUIRED_REASON_INVALID:{prefix}")
        elif record.review_status == "quarantined":
            if not record.quarantine_reason:
                errors.append(f"QUARANTINE_REASON_REQUIRED:{prefix}")
            else:
                quarantine_counts[record.quarantine_reason] += 1
            if record.annotation_status is not None:
                errors.append(f"QUARANTINED_ANNOTATION_STATUS_FORBIDDEN:{prefix}")
    represented = tuple(name for name in DEFECT_NAMES if class_boxes[name])
    return CandidateAudit(
        total_images=len(records),
        accepted_images=accepted_images,
        represented_classes=represented,
        class_boxes=dict(class_boxes),
        quarantine_counts=dict(sorted(quarantine_counts.items())),
        errors=tuple(sorted(errors)),
    )


def assess_license_state(
    *,
    recorded_sha: str,
    current_sha: str,
) -> Literal["verified", "quarantined_license_change"]:
    if not _SHA256.fullmatch(recorded_sha) or not _SHA256.fullmatch(current_sha):
        raise ValueError("LICENSE_SHA256_INVALID")
    return "verified" if recorded_sha == current_sha else "quarantined_license_change"


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit public external FC-BGA candidates.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--json-report", type=Path)
    args = parser.parse_args()
    sources = load_source_registry(args.sources)
    records = load_candidate_manifest(args.manifest, sources)
    report = audit_candidates(records, sources=sources, manifest_root=args.manifest.parent)
    output = json.dumps(asdict(report), indent=2, sort_keys=True)
    if args.json_report:
        args.json_report.parent.mkdir(parents=True, exist_ok=True)
        args.json_report.write_text(output + "\n", encoding="utf-8")
    print(output)
    return 1 if report.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
