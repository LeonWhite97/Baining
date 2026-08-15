from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from typing import Iterable

from PIL import Image, UnidentifiedImageError

try:
    from .contracts import DEFECT_NAMES, INPUT_CONTRACT, REQUIRED_LIGHTS
except ImportError:
    from contracts import DEFECT_NAMES, INPUT_CONTRACT, REQUIRED_LIGHTS


_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MANIFEST_KEYS = {
    "sample_id",
    "group_id",
    "split",
    "input_contract",
    "input_sha256",
    "label_sha256",
    "output_image",
    "output_label",
    "output_sha256",
}


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
    expected_images: set[str] = set()
    expected_labels: set[str] = set()
    root = root.resolve()
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            yield f"MANIFEST_JSON_INVALID:{line_number}"
            continue
        if not isinstance(item, dict):
            yield f"MANIFEST_RECORD_INVALID:{line_number}"
            continue
        sample_id = item.get("sample_id")
        group_id = item.get("group_id")
        split = item.get("split")
        if (
            not isinstance(sample_id, str)
            or not sample_id
            or not isinstance(group_id, str)
            or not group_id
            or split not in {"train", "val", "test"}
        ):
            yield f"MANIFEST_FIELD_INVALID:{line_number}"
            continue
        if sample_id in sample_ids:
            yield f"DUPLICATE_SAMPLE_ID:{sample_id}"
        sample_ids.add(sample_id)
        groups[group_id].add(split)
        if set(item) != _MANIFEST_KEYS:
            yield f"MANIFEST_RECORD_INVALID:{line_number}"
            continue
        if item.get("input_contract") != INPUT_CONTRACT:
            yield f"INPUT_CONTRACT_MISMATCH:{sample_id}"
        input_hashes = item.get("input_sha256")
        if (
            not isinstance(input_hashes, dict)
            or set(input_hashes) != set(REQUIRED_LIGHTS)
            or any(not isinstance(value, str) or not _SHA256.fullmatch(value) for value in input_hashes.values())
        ):
            yield f"INPUT_HASH_INVALID:{sample_id}"
        artifacts = (
            (
                "output_image",
                "output_sha256",
                f"{split}/images/{sample_id}.png",
                expected_images,
                "IMAGE",
            ),
            (
                "output_label",
                "label_sha256",
                f"{split}/labels/{sample_id}.txt",
                expected_labels,
                "LABEL",
            ),
        )
        for path_field, hash_field, expected_relative, expected_set, kind in artifacts:
            relative = item.get(path_field)
            expected_hash = item.get(hash_field)
            if not isinstance(relative, str) or not isinstance(expected_hash, str):
                yield f"MANIFEST_FIELD_INVALID:{sample_id}:{path_field}"
                continue
            path = (root / relative).resolve()
            try:
                path.relative_to(root)
            except ValueError:
                yield f"PATH_OUTSIDE_ROOT:{sample_id}"
                continue
            normalized = path.relative_to(root).as_posix()
            if normalized != expected_relative:
                yield f"MANIFEST_PATH_MISMATCH:{sample_id}:{path_field}"
            if normalized in expected_set:
                yield f"DUPLICATE_MANIFEST_ARTIFACT:{normalized}"
            expected_set.add(normalized)
            if not path.is_file():
                yield f"{kind}_MISSING:{sample_id}"
            elif not _SHA256.fullmatch(expected_hash) or _hash(path) != expected_hash:
                yield f"{kind}_HASH_MISMATCH:{sample_id}"
    actual_images = {
        path.relative_to(root).as_posix()
        for split in ("train", "val", "test")
        for path in (root / split / "images").rglob("*")
        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
    }
    actual_labels = {
        path.relative_to(root).as_posix()
        for split in ("train", "val", "test")
        for path in (root / split / "labels").rglob("*.txt")
        if path.is_file()
    }
    for relative in sorted(actual_images - expected_images):
        yield f"UNMANIFESTED_IMAGE:{relative}"
    for relative in sorted(actual_labels - expected_labels):
        yield f"UNMANIFESTED_LABEL:{relative}"
    for group_id, splits in groups.items():
        if len(splits) > 1:
            yield f"GROUP_LEAKAGE:{group_id}:{','.join(sorted(splits))}"


def _unsupported_tree_files(root: Path) -> Iterable[str]:
    for split in ("train", "val", "test"):
        directories = (
            (root / split / "images", _IMAGE_SUFFIXES, "IMAGE"),
            (root / split / "labels", {".txt"}, "LABEL"),
        )
        for directory, suffixes, kind in directories:
            if not directory.is_dir():
                continue
            for path in directory.rglob("*"):
                if (
                    path.is_file()
                    and path.name != ".gitkeep"
                    and path.suffix.lower() not in suffixes
                ):
                    relative = path.relative_to(root).as_posix()
                    yield f"UNSUPPORTED_{kind}_FILE:{relative}"


def validate_dataset(
    root: Path,
    class_names: tuple[str, ...],
    manifest: Path | None,
    *,
    require_nonempty_splits: bool = False,
) -> ValidationReport:
    root = root.resolve()
    errors: list[str] = []
    images = boxes = empty_labels = 0
    split_images: Counter[str] = Counter()
    class_boxes: Counter[str] = Counter()
    errors.extend(_unsupported_tree_files(root))
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
        if require_nonempty_splits and not image_by_stem:
            errors.append(f"EMPTY_SPLIT:{split}")
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
