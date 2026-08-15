from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from types import MappingProxyType

from PIL import Image, UnidentifiedImageError

try:
    from .contracts import REQUIRED_LIGHTS
except ImportError:
    from contracts import REQUIRED_LIGHTS


SUPPORTED_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png"}


@dataclass(frozen=True, slots=True)
class MockCaptureReport:
    samples: int
    split_counts: MappingProxyType[str, int]
    manifest: Path


def _input_images(source_root: Path) -> tuple[Path, ...]:
    if not source_root.is_dir():
        raise ValueError(f"SOURCE_DIR_INVALID: {source_root}")
    return tuple(
        sorted(
            path
            for path in source_root.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )
    )


def _split_for(index: int, total: int, train_ratio: float, val_ratio: float) -> str:
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    if total >= 3:
        train_count = max(1, min(train_count, total - 2))
        val_count = max(1, min(val_count, total - train_count - 1))
    elif total == 2:
        train_count = 1
        val_count = 0
    else:
        train_count = 1
        val_count = 0
    if index < train_count:
        return "train"
    if index < train_count + val_count:
        return "val"
    return "test"


def _prepare_output(output_root: Path, *, force: bool) -> None:
    if output_root.exists():
        if not output_root.is_dir() or output_root.is_symlink():
            raise ValueError("OUTPUT_INVALID")
        if any(output_root.iterdir()):
            if not force:
                raise ValueError("OUTPUT_NOT_EMPTY")
            shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)


def _save_png_copy(source: Path, destination: Path) -> None:
    try:
        with Image.open(source) as image:
            image.load()
            image.convert("RGB").save(destination, format="PNG")
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError(f"IMAGE_DECODE_FAILED: {source.name}") from exc


def create_mock_capture_dataset(
    source_root: Path,
    output_root: Path,
    *,
    prefix: str = "MOCK",
    group_id: str = "mock_{sample_id}",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    force: bool = False,
) -> MockCaptureReport:
    if not prefix or not prefix.replace("_", "").replace("-", "").isalnum():
        raise ValueError("PREFIX_INVALID")
    if not group_id.strip():
        raise ValueError("GROUP_ID_INVALID")
    if not 0 < train_ratio < 1 or not 0 <= val_ratio < 1 or train_ratio + val_ratio >= 1:
        raise ValueError("SPLIT_RATIO_INVALID")
    images = _input_images(source_root)
    if not images:
        raise ValueError("SOURCE_IMAGES_EMPTY")
    _prepare_output(output_root, force=force)
    raw_root = output_root / "raw"
    label_root = output_root / "annotations"
    raw_root.mkdir()
    label_root.mkdir()
    records: list[dict[str, object]] = []
    split_counts: Counter[str] = Counter()
    for index, source in enumerate(images):
        sample_id = f"{prefix}{index + 1:04d}"
        split = _split_for(index, len(images), train_ratio, val_ratio)
        light_paths: dict[str, str] = {}
        for light in REQUIRED_LIGHTS:
            destination = raw_root / f"{sample_id}_{light}.png"
            _save_png_copy(source, destination)
            light_paths[light] = destination.relative_to(output_root).as_posix()
        label_path = label_root / f"{sample_id}.txt"
        label_path.write_text("", encoding="utf-8")
        sample_group_id = group_id.strip().format(sample_id=sample_id)
        records.append(
            {
                "sample_id": sample_id,
                "group_id": sample_group_id,
                "split": split,
                "images": light_paths,
                "label": label_path.relative_to(output_root).as_posix(),
            }
        )
        split_counts[split] += 1
    manifest = output_root / "source.jsonl"
    manifest.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    return MockCaptureReport(
        samples=len(records),
        split_counts=MappingProxyType(dict(split_counts)),
        manifest=manifest,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a mock four-light FC-BGA capture manifest from existing images."
    )
    parser.add_argument("source", type=Path, help="Directory containing existing images.")
    parser.add_argument("output", type=Path, help="Output mock capture directory.")
    parser.add_argument("--prefix", default="MOCK")
    parser.add_argument(
        "--group-id",
        default="mock_{sample_id}",
        help="Group id value or template. The default isolates each mock sample.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    report = create_mock_capture_dataset(
        args.source,
        args.output,
        prefix=args.prefix,
        group_id=args.group_id,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        force=args.force,
    )
    print(
        json.dumps(
            {
                "samples": report.samples,
                "split_counts": dict(report.split_counts),
                "manifest": str(report.manifest),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
