from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
from pathlib import Path


_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
_SPLIT_PRIORITY = {"test": 0, "val": 1, "train": 2}


@dataclass(frozen=True, slots=True)
class DuplicateGroup:
    image_sha256: str
    keep_image: Path
    remove_images: tuple[Path, ...]
    label_conflict: bool


@dataclass(frozen=True, slots=True)
class DeduplicationReport:
    root: Path
    groups: tuple[DuplicateGroup, ...]
    redundant_images: int
    conflicts: int
    removed_images: int = 0


def _hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _label_path(image: Path) -> Path:
    return image.parent.parent / "labels" / f"{image.stem}.txt"


def _normalized_label(image: Path) -> str | None:
    label = _label_path(image)
    if not label.is_file():
        return None
    return "\n".join(" ".join(line.split()) for line in label.read_text(encoding="utf-8-sig").splitlines() if line.strip())


def _sort_key(root: Path, image: Path) -> tuple[int, str]:
    relative = image.relative_to(root)
    split = relative.parts[0] if relative.parts else ""
    return _SPLIT_PRIORITY.get(split, 99), relative.as_posix()


def audit_duplicates(root: Path) -> DeduplicationReport:
    root = root.resolve()
    by_hash: dict[str, list[Path]] = {}
    for path in root.rglob("*"):
        if path.is_file() and path.parent.name == "images" and path.suffix.lower() in _IMAGE_SUFFIXES:
            by_hash.setdefault(_hash(path), []).append(path)
    groups: list[DuplicateGroup] = []
    for image_hash, paths in sorted(by_hash.items()):
        if len(paths) < 2:
            continue
        ordered = sorted(paths, key=lambda item: _sort_key(root, item))
        labels = {_normalized_label(path) for path in ordered}
        groups.append(
            DuplicateGroup(
                image_sha256=image_hash,
                keep_image=ordered[0],
                remove_images=tuple(ordered[1:]),
                label_conflict=len(labels) != 1,
            )
        )
    return DeduplicationReport(
        root=root,
        groups=tuple(groups),
        redundant_images=sum(len(group.remove_images) for group in groups),
        conflicts=sum(group.label_conflict for group in groups),
    )


def apply_duplicate_report(report: DeduplicationReport) -> DeduplicationReport:
    if report.conflicts:
        raise ValueError("LABEL_CONFLICT: duplicate images have different labels")
    removed = 0
    for group in report.groups:
        for image in group.remove_images:
            label = _label_path(image)
            image.unlink(missing_ok=False)
            label.unlink(missing_ok=True)
            removed += 1
    return replace(report, removed_images=removed)


def _json_report(report: DeduplicationReport) -> dict[str, object]:
    return {
        "root": str(report.root),
        "redundant_images": report.redundant_images,
        "conflicts": report.conflicts,
        "removed_images": report.removed_images,
        "groups": [
            {
                "image_sha256": group.image_sha256,
                "keep_image": str(group.keep_image),
                "remove_images": [str(path) for path in group.remove_images],
                "label_conflict": group.label_conflict,
            }
            for group in report.groups
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit or apply exact-content YOLO deduplication.")
    parser.add_argument("root", type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json-report", type=Path)
    args = parser.parse_args()
    report = audit_duplicates(args.root)
    if args.apply:
        report = apply_duplicate_report(report)
    output = json.dumps(_json_report(report), sort_keys=True, indent=2)
    if args.json_report:
        args.json_report.write_text(output + "\n", encoding="utf-8")
    print(output)
    return 1 if report.conflicts else 0


if __name__ == "__main__":
    raise SystemExit(main())

