from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Iterable

from PIL import Image, UnidentifiedImageError

try:
    from .contracts import DEFECT_NAMES
except ImportError:
    from contracts import DEFECT_NAMES


_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True, slots=True)
class ValidationReport:
    images: int
    boxes: int
    empty_labels: int
    split_images: dict[str, int]
    class_boxes: dict[str, int]
    errors: tuple[str, ...]


def _hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_label(path: Path, class_names: tuple[str, ...]) -> tuple[int, list[str], Counter[str]]:
    errors: list[str] = []
    boxes = 0
    class_boxes: Counter[str] = Counter()
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError:
        return 0, [f"LABEL_UNAVAILABLE:{path.as_posix()}"], class_boxes
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"LABEL_FORMAT_INVALID:{path.as_posix()}:{line_number}")
            continue
        try:
            class_id = int(parts[0])
            x, y, width, height = (float(value) for value in parts[1:])
        except ValueError:
            errors.append(f"LABEL_FORMAT_INVALID:{path.as_posix()}:{line_number}")
            continue
        values = (x, y, width, height)
        if not 0 <= class_id < len(class_names):
            errors.append(f"CLASS_ID_OUT_OF_RANGE:{path.as_posix()}:{line_number}")
        elif (
            not all(math.isfinite(value) for value in values)
            or not 0 <= x <= 1
            or not 0 <= y <= 1
            or not 0 < width <= 1
            or not 0 < height <= 1
        ):
            errors.append(f"BOX_VALUE_INVALID:{path.as_posix()}:{line_number}")
        else:
            boxes += 1
            class_boxes[class_names[class_id]] += 1
    return boxes, errors, class_boxes


def _manifest_errors(root: Path, manifest: Path) -> Iterable[str]:
    groups: defaultdict[str, set[str]] = defaultdict(set)
    sample_ids: set[str] = set()
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            yield f"MANIFEST_JSON_INVALID:{line_number}"
            continue
        sample_id = item.get("sample_id")
        group_id = item.get("group_id")
        split = item.get("split")
        if sample_id in sample_ids:
            yield f"DUPLICATE_SAMPLE_ID:{sample_id}"
        if isinstance(sample_id, str):
            sample_ids.add(sample_id)
        if isinstance(group_id, str) and split in {"train", "val", "test"}:
            groups[group_id].add(split)
        output_image = item.get("output_image")
        expected_hash = item.get("output_sha256")
        if isinstance(output_image, str) and isinstance(expected_hash, str):
            path = (root / output_image).resolve()
            try:
                path.relative_to(root.resolve())
            except ValueError:
                yield f"PATH_OUTSIDE_ROOT:{sample_id}"
                continue
            if not path.is_file() or _hash(path) != expected_hash:
                yield f"HASH_MISMATCH:{sample_id}"
    for group_id, splits in groups.items():
        if len(splits) > 1:
            yield f"GROUP_LEAKAGE:{group_id}:{','.join(sorted(splits))}"


def validate_dataset(
    root: Path,
    class_names: tuple[str, ...],
    manifest: Path | None,
) -> ValidationReport:
    root = root.resolve()
    errors: list[str] = []
    images = boxes = empty_labels = 0
    split_images: Counter[str] = Counter()
    class_boxes: Counter[str] = Counter()
    for split in ("train", "val", "test"):
        image_dir = root / split / "images"
        label_dir = root / split / "labels"
        image_by_stem = {
            path.stem: path
            for path in image_dir.glob("*")
            if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
        } if image_dir.is_dir() else {}
        label_by_stem = {path.stem: path for path in label_dir.glob("*.txt")} if label_dir.is_dir() else {}
        for stem in sorted(image_by_stem):
            image_path = image_by_stem[stem]
            try:
                with Image.open(image_path) as image:
                    image.verify()
            except (OSError, UnidentifiedImageError):
                errors.append(f"IMAGE_DECODE_FAILED:{image_path.relative_to(root).as_posix()}")
            images += 1
            split_images[split] += 1
            label_path = label_by_stem.get(stem)
            if label_path is None:
                errors.append(f"MISSING_LABEL:{split}:{stem}")
                continue
            label_boxes, label_errors, counts = _validate_label(label_path, class_names)
            boxes += label_boxes
            class_boxes.update(counts)
            errors.extend(label_errors)
            if label_path.stat().st_size == 0:
                empty_labels += 1
        for stem in sorted(set(label_by_stem) - set(image_by_stem)):
            errors.append(f"MISSING_IMAGE:{split}:{stem}")
    if manifest is not None:
        if not manifest.is_file():
            errors.append(f"MANIFEST_UNAVAILABLE:{manifest.name}")
        else:
            errors.extend(_manifest_errors(root, manifest))
    return ValidationReport(
        images=images,
        boxes=boxes,
        empty_labels=empty_labels,
        split_images=dict(sorted(split_images.items())),
        class_boxes=dict(sorted(class_boxes.items())),
        errors=tuple(sorted(errors)),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate an FC-BGA YOLO dataset.")
    parser.add_argument("root", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--json-report", type=Path)
    args = parser.parse_args()
    report = validate_dataset(args.root, DEFECT_NAMES, args.manifest)
    output = json.dumps(asdict(report), sort_keys=True, indent=2)
    if args.json_report:
        args.json_report.write_text(output + "\n", encoding="utf-8")
    print(output)
    return 1 if report.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())

