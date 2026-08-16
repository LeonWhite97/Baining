from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import shutil

try:
    from .model_metadata import sha256_file
    from .public_external_manifest import CandidateRecord, load_source_registry
except ImportError:
    from model_metadata import sha256_file
    from public_external_manifest import CandidateRecord, load_source_registry


_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class CandidatePreparationReport:
    source_images: int
    unique_images: int
    exact_duplicates: int
    manifest: Path
    review_root: Path


def _load_source_manifest(path: Path) -> dict[str, object]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError) as exc:
        raise ValueError("SOURCE_MANIFEST_INVALID") from exc
    required = {"source_url", "version", "license", "files"}
    if not isinstance(document, dict) or not required.issubset(document):
        raise ValueError("SOURCE_MANIFEST_INVALID")
    files = document["files"]
    if not isinstance(files, dict) or any(
        not isinstance(name, str) or not isinstance(digest, str)
        for name, digest in files.items()
    ):
        raise ValueError("SOURCE_MANIFEST_INVALID")
    return document


def _source_group_id(source_id: str, filename: str) -> str:
    original_stem = Path(filename).stem.split(".rf.", maxsplit=1)[0].casefold()
    return f"{source_id}:{original_stem}"


def _write_manifest(path: Path, records: tuple[CandidateRecord, ...]) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    content = "".join(json.dumps(asdict(record), sort_keys=True) + "\n" for record in records)
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def prepare_candidates(
    source_root: Path,
    source_manifest: Path,
    source_registry: Path,
    destination: Path,
    *,
    source_id: str,
) -> CandidatePreparationReport:
    root = source_root.resolve()
    document = _load_source_manifest(source_manifest)
    sources = {source.source_id: source for source in load_source_registry(source_registry)}
    source = sources.get(source_id)
    if source is None:
        raise ValueError("SOURCE_UNREGISTERED")
    if (
        document["source_url"] != source.url
        or str(document["version"]) != source.version
        or document["license"] != source.license_name
    ):
        raise ValueError("SOURCE_REGISTRY_MISMATCH")

    image_items: list[tuple[str, Path, str]] = []
    for relative, expected_hash in sorted(document["files"].items()):
        normalized = relative.replace("\\", "/")
        if "/images/" not in f"/{normalized}":
            continue
        suffix = Path(normalized).suffix.lower()
        if suffix not in _IMAGE_SUFFIXES:
            raise ValueError(f"UNSUPPORTED_EXTENSION:{relative}")
        if not _SHA256.fullmatch(expected_hash):
            raise ValueError(f"SOURCE_HASH_INVALID:{relative}")
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"SOURCE_PATH_ESCAPE:{relative}") from exc
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ValueError(f"SOURCE_HASH_MISMATCH:{relative}")
        image_items.append((relative, path, expected_hash))
    if not image_items:
        raise ValueError("SOURCE_IMAGES_UNAVAILABLE")

    unique: dict[str, tuple[str, Path]] = {}
    for relative, path, digest in image_items:
        unique.setdefault(digest, (relative, path))
    destination.mkdir(parents=True, exist_ok=True)
    images_root = destination / "images"
    images_root.mkdir(exist_ok=True)
    records: list[CandidateRecord] = []
    for digest, (relative, source_path) in sorted(unique.items()):
        sample_id = f"public-{digest[:16]}"
        output = images_root / f"{sample_id}{source_path.suffix.lower()}"
        if output.exists():
            if not output.is_file() or sha256_file(output) != digest:
                raise ValueError(f"OUTPUT_CONTENT_CONFLICT:{output.name}")
        else:
            shutil.copy2(source_path, output)
        records.append(
            CandidateRecord(
                sample_id=sample_id,
                source_group_id=_source_group_id(source_id, Path(relative).name),
                source_id=source_id,
                original_filename=Path(relative).name,
                image_path=output.relative_to(destination).as_posix(),
                image_sha256=digest,
                label_path=None,
                review_status="review_required",
                annotation_status=None,
                accepted_classes=(),
                quarantine_reason=None,
            )
        )
    manifest = destination / "candidates.jsonl"
    _write_manifest(manifest, tuple(records))
    return CandidatePreparationReport(
        source_images=len(image_items),
        unique_images=len(records),
        exact_duplicates=len(image_items) - len(records),
        manifest=manifest,
        review_root=destination,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare public FC-BGA candidates for manual review.")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-registry", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--source-id", required=True)
    args = parser.parse_args()
    report = prepare_candidates(
        args.source_root,
        args.source_manifest,
        args.source_registry,
        args.destination,
        source_id=args.source_id,
    )
    print(json.dumps({**asdict(report), "manifest": str(report.manifest), "review_root": str(report.review_root)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
