from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from types import MappingProxyType
from typing import Mapping

from PIL import Image, UnidentifiedImageError

try:
    from .contracts import DEFECT_NAMES, INPUT_CONTRACT, REQUIRED_LIGHTS
    from .preprocessing import stack_rgb_grayscale
except ImportError:
    from contracts import DEFECT_NAMES, INPUT_CONTRACT, REQUIRED_LIGHTS
    from preprocessing import stack_rgb_grayscale


_SAMPLE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SPLITS = {"train", "val", "test"}


@dataclass(frozen=True, slots=True)
class SourceSample:
    sample_id: str
    group_id: str
    split: str
    images: Mapping[str, Path]
    label: Path


@dataclass(frozen=True, slots=True)
class ConversionReport:
    samples: int
    split_counts: Mapping[str, int]
    output_manifest: Path


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_inside(root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("MANIFEST_FIELD_INVALID: paths must be non-empty strings")
    candidate = (root / value).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("PATH_OUTSIDE_ROOT") from exc
    if not candidate.is_file():
        raise ValueError(f"SOURCE_FILE_MISSING: {candidate.name}")
    return candidate


def parse_source_manifest(path: Path) -> tuple[SourceSample, ...]:
    root = path.resolve().parent
    seen: set[str] = set()
    samples: list[SourceSample] = []
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError as exc:
        raise ValueError(f"MANIFEST_UNAVAILABLE: {path.name}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"MANIFEST_JSON_INVALID: line {line_number}") from exc
        if not isinstance(item, dict):
            raise ValueError(f"MANIFEST_RECORD_INVALID: line {line_number}")
        sample_id = item.get("sample_id")
        group_id = item.get("group_id")
        split = item.get("split")
        images = item.get("images")
        if not isinstance(sample_id, str) or not _SAMPLE_ID.fullmatch(sample_id):
            raise ValueError(f"SAMPLE_ID_INVALID: line {line_number}")
        if sample_id in seen:
            raise ValueError(f"DUPLICATE_SAMPLE_ID: {sample_id}")
        if not isinstance(group_id, str) or not group_id.strip():
            raise ValueError(f"GROUP_ID_INVALID: {sample_id}")
        if split not in _SPLITS:
            raise ValueError(f"SPLIT_INVALID: {sample_id}")
        if not isinstance(images, dict) or set(images) != set(REQUIRED_LIGHTS):
            raise ValueError(f"LIGHT_SET_INVALID: {sample_id}")
        resolved_images = {
            light: _resolve_inside(root, images[light]) for light in REQUIRED_LIGHTS
        }
        label = _resolve_inside(root, item.get("label"))
        samples.append(
            SourceSample(
                sample_id=sample_id,
                group_id=group_id.strip(),
                split=split,
                images=MappingProxyType(resolved_images),
                label=label,
            )
        )
        seen.add(sample_id)
    if not samples:
        raise ValueError("MANIFEST_EMPTY")
    return tuple(samples)


def _validate_label(path: Path) -> None:
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"LABEL_FORMAT_INVALID: {path.name}:{line_number}")
        try:
            class_id = int(parts[0])
            values = tuple(float(value) for value in parts[1:])
        except ValueError as exc:
            raise ValueError(f"LABEL_FORMAT_INVALID: {path.name}:{line_number}") from exc
        x, y, width, height = values
        if (
            not 0 <= class_id < len(DEFECT_NAMES)
            or not all(math.isfinite(value) for value in values)
            or not 0 <= x <= 1
            or not 0 <= y <= 1
            or not 0 < width <= 1
            or not 0 < height <= 1
        ):
            raise ValueError(f"LABEL_VALUE_INVALID: {path.name}:{line_number}")


def _image_size(path: Path) -> tuple[int, int]:
    try:
        with Image.open(path) as image:
            if image.format not in {"JPEG", "PNG"}:
                raise ValueError(f"IMAGE_FORMAT_INVALID: {path.name}")
            image.load()
            return image.size
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError(f"IMAGE_DECODE_FAILED: {path.name}") from exc


def convert_manifest(manifest_path: Path, output_root: Path) -> ConversionReport:
    samples = parse_source_manifest(manifest_path)
    records: list[dict[str, object]] = []
    counts: Counter[str] = Counter()
    for sample in samples:
        _validate_label(sample.label)
        sizes = {_image_size(sample.images[light]) for light in REQUIRED_LIGHTS}
        if len(sizes) != 1:
            raise ValueError(f"IMAGE_SIZE_MISMATCH: {sample.sample_id}")
        stacked = stack_rgb_grayscale(
            sample.images["R"], sample.images["G"], sample.images["B"]
        )
        image_path = output_root / sample.split / "images" / f"{sample.sample_id}.png"
        label_path = output_root / sample.split / "labels" / f"{sample.sample_id}.txt"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        image_tmp = image_path.with_suffix(".png.tmp")
        label_tmp = label_path.with_suffix(".txt.tmp")
        try:
            stacked.save(image_tmp, format="PNG")
            label_tmp.write_bytes(sample.label.read_bytes())
            image_tmp.replace(image_path)
            label_tmp.replace(label_path)
        finally:
            image_tmp.unlink(missing_ok=True)
            label_tmp.unlink(missing_ok=True)
        records.append(
            {
                "sample_id": sample.sample_id,
                "group_id": sample.group_id,
                "split": sample.split,
                "input_contract": INPUT_CONTRACT,
                "input_sha256": {
                    light: _sha256_file(sample.images[light]) for light in REQUIRED_LIGHTS
                },
                "label_sha256": _sha256_file(label_path),
                "output_image": image_path.relative_to(output_root).as_posix(),
                "output_label": label_path.relative_to(output_root).as_posix(),
                "output_sha256": _sha256_file(image_path),
            }
        )
        counts[sample.split] += 1
    output_manifest = output_root / "manifest.jsonl"
    manifest_tmp = output_manifest.with_suffix(".jsonl.tmp")
    manifest_tmp.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    manifest_tmp.replace(output_manifest)
    return ConversionReport(len(samples), MappingProxyType(dict(counts)), output_manifest)


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert a four-light FC-BGA JSONL manifest to YOLO format.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = convert_manifest(args.manifest, args.output)
    print(json.dumps({"samples": report.samples, "manifest": str(report.output_manifest)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
