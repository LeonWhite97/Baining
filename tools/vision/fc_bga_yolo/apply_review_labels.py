"""Apply exported YOLO label files into the public-external candidate manifest.

This closes the manual loop that `review_progress.py` / `build_b0_version.py`
expect: after the human draws boxes in an annotation tool (makesense.ai,
LabelImg, Label Studio) and exports YOLO `.txt` files, this script

1. matches each label file to a candidate by `{sample_id}.txt`,
2. validates every box (same routine the B0 gate uses),
3. derives the represented defect classes,
4. writes `review_status`, `annotation_status`, `accepted_classes`,
   `label_path` back into `candidates.jsonl`, and
5. copies the label into `review/labels/` (which is git-tracked).

It removes the need to hand-edit JSONL.

Class-index rules
-----------------
YOLO `.txt` files always store integer class ids, never names. Two cases:

* Default — the tool added the 7 classes **in `DEFECT_NAMES` order**
  (0 BALL_BRIDGE, 1 MISSING_BALL, ... 6 FOREIGN_MATERIAL). Nothing else
  needed.
* Different order — pass `--class-map map.json`, e.g.
  `{"0": "MISSING_BALL", "1": "BALL_BRIDGE", ...}` translating the tool's
  index -> a defect name. The script rewrites indices to `DEFECT_NAMES`
  order on copy, so the stored file always matches the canonical order.

`accepted_classes` is always emitted in `DEFECT_NAMES` order, exactly as the
B0 gate's `audit_candidates` compares it.

Safety
------
* `candidates.jsonl` is backed up to `candidates.jsonl.bak` before any write.
* Re-running is idempotent: only candidates that have a matching label file
  are touched.
* `--dry-run` prints the planned changes without writing anything.
* Candidate records whose label fails validation are reported and left as
  `review_required` (never silently accepted).

Quarantining a sample is still a manual edit in `candidates.jsonl` — this
script only handles accepted-with-label.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from tempfile import NamedTemporaryFile

sys.path.insert(0, str(Path(__file__).resolve().parent))

from contracts import DEFECT_NAMES  # noqa: E402
from public_external_manifest import (  # noqa: E402
    CandidateRecord,
    load_candidate_manifest,
    load_source_registry,
)
from validate_yolo_dataset import _validate_label  # noqa: E402

ANNOTATION_STATUS = "provisional_human_reviewed_poc"


def _parse_box_class(idx_str: str, class_map: dict[str, str] | None) -> str:
    """Translate a raw label index token to a canonical defect name."""
    if class_map:
        name = class_map.get(idx_str)
        if name is None:
            raise ValueError(f"index {idx_str!r} missing from --class-map")
        if name not in DEFECT_NAMES:
            raise ValueError(f"--class-map target {name!r} is not a defect name")
        return name
    try:
        i = int(idx_str)
    except ValueError as exc:
        raise ValueError(
            f"label index {idx_str!r} is not an int (use --class-map if the "
            "tool used a different class order)"
        ) from exc
    if i < 0 or i >= len(DEFECT_NAMES):
        raise ValueError(f"label index {i} out of range [0,{len(DEFECT_NAMES) - 1}]")
    return DEFECT_NAMES[i]


def _remap_label_text(text: str, class_map: dict[str, str] | None) -> tuple[str, set[str]]:
    """Rewrite a YOLO label body to canonical indices; return (text, present names)."""
    out_lines: list[str] = []
    present: set[str] = set()
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            raise ValueError(f"malformed box line: {line!r}")
        name = _parse_box_class(parts[0], class_map)
        present.add(name)
        out_lines.append(" ".join([str(DEFECT_NAMES.index(name)), *parts[1:]]))
    body = "\n".join(out_lines)
    return (body + "\n") if body else "", present


def _ordered_classes(present: set[str]) -> tuple[str, ...]:
    """Emit present classes in DEFECT_NAMES order (matches audit_candidates)."""
    return tuple(name for name in DEFECT_NAMES if name in present)


def _load_class_map(path: Path | None) -> dict[str, str] | None:
    if path is None:
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"CLASS_MAP_INVALID:{path}") from exc
    if not isinstance(raw, dict):
        raise ValueError("CLASS_MAP_INVALID: top-level must be an object")
    return {str(k): str(v) for k, v in raw.items()}


def apply_labels(
    manifest: Path,
    sources: Path,
    label_dir: Path,
    class_map_path: Path | None,
    *,
    dry_run: bool,
) -> int:
    src_records = load_source_registry(sources)
    records = load_candidate_manifest(manifest, src_records)
    class_map = _load_class_map(class_map_path)

    by_id = {r.sample_id: r for r in records}
    review_root = manifest.parent
    labels_root = review_root / "labels"
    labels_root.mkdir(parents=True, exist_ok=True)

    updated = 0
    skipped_no_label = 0
    failed: list[str] = []

    # Build the new record tuple; untouched records are reused verbatim.
    new_records: list[CandidateRecord] = []
    for record in records:
        label_src = label_dir / f"{record.sample_id}.txt"
        if not label_src.is_file():
            skipped_no_label += 1
            new_records.append(record)
            continue

        try:
            body, present = _remap_label_text(
                label_src.read_text(encoding="utf-8-sig"), class_map
            )
        except ValueError as exc:
            failed.append(f"{record.sample_id}: LABEL_PARSE_ERROR {exc}")
            new_records.append(record)
            continue

        target = labels_root / f"{record.sample_id}.txt"
        if dry_run:
            # Validate the in-memory body via a throwaway temp file so the
            # dry run leaves no side effects.
            with NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as tf:
                tf.write(body)
                tmp_path = Path(tf.name)
            boxes, errors, _counts = _validate_label(tmp_path, DEFECT_NAMES)
            tmp_path.unlink(missing_ok=True)
        else:
            target.write_text(body, encoding="utf-8")
            boxes, errors, _counts = _validate_label(target, DEFECT_NAMES)
        if errors:
            failed.append(f"{record.sample_id}: " + "; ".join(errors))
            new_records.append(record)
            continue

        ordered = _ordered_classes(present)
        new_records.append(
            CandidateRecord(
                sample_id=record.sample_id,
                source_group_id=record.source_group_id,
                source_id=record.source_id,
                original_filename=record.original_filename,
                image_path=record.image_path,
                image_sha256=record.image_sha256,
                label_path=f"labels/{record.sample_id}.txt",
                review_status="accepted",
                annotation_status=ANNOTATION_STATUS,
                accepted_classes=ordered,
                quarantine_reason=None,
            )
        )
        updated += 1

    _print_summary(updated, skipped_no_label, failed, dry_run)

    if dry_run or failed and updated == 0:
        # Never write if nothing valid changed.
        if dry_run:
            return 1 if failed else 0
        return 1

    backup = manifest.with_suffix(manifest.suffix + ".bak")
    shutil.copy2(manifest, backup)
    lines = [json.dumps(asdict(r)) for r in new_records]
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[write] {manifest}  (backup -> {backup.name})")
    return 1 if failed else 0


def _print_summary(updated: int, skipped: int, failed: list[str], dry_run: bool) -> None:
    tag = "[dry-run] " if dry_run else ""
    print(f"{tag}updated accepted: {updated}")
    print(f"{tag}skipped (no label file): {skipped}")
    if failed:
        print(f"{tag}failed validation ({len(failed)}):")
        for item in failed:
            print(f"  - {item}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply exported YOLO labels into the candidate manifest."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).resolve().parent.parent.parent.parent
        / "data/external/fc_bga_public_external/review/candidates.jsonl",
        help="path to candidates.jsonl",
    )
    parser.add_argument(
        "--sources",
        type=Path,
        default=Path(__file__).resolve().parent.parent.parent.parent
        / "data/external/fc_bga_public_external/sources.json",
        help="path to sources.json (needed to validate source_ids)",
    )
    parser.add_argument(
        "--label-dir",
        type=Path,
        required=True,
        help="directory holding exported {sample_id}.txt label files",
    )
    parser.add_argument(
        "--class-map",
        type=Path,
        default=None,
        help="optional JSON {tool_index: DEFECT_NAME} when the tool used a "
        "different class order",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print planned changes, write nothing"
    )
    args = parser.parse_args()
    return apply_labels(
        args.manifest,
        args.sources,
        args.label_dir,
        args.class_map,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
