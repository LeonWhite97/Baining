from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import shutil
import tempfile
from typing import Literal, Mapping

import yaml

try:
    from .contracts import DEFECT_NAMES
    from .model_metadata import sha256_file
    from .public_external_manifest import (
        CandidateRecord,
        load_candidate_manifest,
        load_source_registry,
    )
    from .validate_yolo_dataset import _validate_label
except ImportError:
    from contracts import DEFECT_NAMES
    from model_metadata import sha256_file
    from public_external_manifest import CandidateRecord, load_candidate_manifest, load_source_registry
    from validate_yolo_dataset import _validate_label


Stage = Literal["B0", "B1"]
Split = Literal["train", "val", "test"]
_SPLITS: tuple[Split, ...] = ("train", "val", "test")
_PROPORTIONS: Mapping[Split, float] = {"train": 0.70, "val": 0.15, "test": 0.15}


@dataclass(frozen=True, slots=True)
class RevisionGate:
    stage: Stage
    status: Literal["ready", "blocked_data"]
    accepted_images: int
    represented_classes: tuple[str, ...]
    train_boxes: Mapping[str, int]
    test_boxes: Mapping[str, int]
    split_images: Mapping[str, int]
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PublishedRevision:
    version: str
    root: Path
    manifest: Path
    assignments: Path
    data_yaml: Path
    revision_json: Path
    manifest_sha256: str


def _seeded_rank(seed: int, group_id: str) -> str:
    return sha256(f"{seed}:{group_id}".encode("utf-8")).hexdigest()


def assign_group_stratified_v1(
    records: tuple[CandidateRecord, ...],
    *,
    seed: int = 42,
) -> Mapping[str, Split]:
    if not records:
        return {}
    groups: defaultdict[str, list[CandidateRecord]] = defaultdict(list)
    for record in records:
        if not record.source_group_id or not record.sample_id:
            raise ValueError("REVISION_RECORD_INVALID")
        groups[record.source_group_id].append(record)
    class_group_frequency: Counter[str] = Counter()
    for group_records in groups.values():
        for name in set(name for record in group_records for name in record.accepted_classes):
            class_group_frequency[name] += 1
    ordered_groups = sorted(
        groups,
        key=lambda group_id: (
            min(
                (class_group_frequency[name] for record in groups[group_id] for name in record.accepted_classes),
                default=len(groups),
            ),
            _seeded_rank(seed, group_id),
        ),
    )
    total_images = len(records)
    image_targets = {split: total_images * _PROPORTIONS[split] for split in _SPLITS}
    class_targets = {
        split: {
            name: class_group_frequency[name] * _PROPORTIONS[split]
            for name in DEFECT_NAMES
        }
        for split in _SPLITS
    }
    current_images: Counter[str] = Counter()
    current_classes: dict[str, Counter[str]] = {split: Counter() for split in _SPLITS}
    group_split: dict[str, Split] = {}
    for group_id in ordered_groups:
        group_records = groups[group_id]
        group_size = len(group_records)
        group_classes = set(name for record in group_records for name in record.accepted_classes)
        feasible = [
            split
            for split in _SPLITS
            if current_images[split] + group_size <= math.ceil(image_targets[split])
        ]
        candidates = feasible or list(_SPLITS)

        def score(split: Split) -> tuple[float, float, float, int]:
            image_target = max(image_targets[split], 1.0)
            image_deficit = max(image_targets[split] - current_images[split], 0.0) / image_target
            class_deficits = [
                max(class_targets[split][name] - current_classes[split][name], 0.0)
                / max(class_targets[split][name], 1.0)
                for name in group_classes
            ]
            class_deficit = sum(class_deficits) / len(class_deficits) if class_deficits else 0.0
            return class_deficit, image_deficit, -current_images[split], -_SPLITS.index(split)

        selected = max(candidates, key=score)
        group_split[group_id] = selected
        current_images[selected] += group_size
        current_classes[selected].update(group_classes)
    return {
        record.sample_id: group_split[record.source_group_id]
        for record in records
    }


def evaluate_revision_gate(
    records: tuple[CandidateRecord, ...],
    stage: Stage,
    *,
    manifest_root: Path,
    assignments: Mapping[str, Split] | None = None,
) -> RevisionGate:
    if stage not in {"B0", "B1"}:
        raise ValueError("REVISION_STAGE_INVALID")
    accepted = tuple(record for record in records if record.review_status == "accepted")
    selected = assignments or assign_group_stratified_v1(accepted, seed=42)
    reasons: list[str] = []
    split_images: Counter[str] = Counter()
    split_boxes: dict[str, Counter[str]] = {split: Counter() for split in _SPLITS}
    groups: defaultdict[str, set[str]] = defaultdict(set)
    for record in accepted:
        split = selected.get(record.sample_id)
        if split not in _SPLITS:
            reasons.append(f"ASSIGNMENT_INVALID:{record.sample_id}")
            continue
        split_images[split] += 1
        groups[record.source_group_id].add(split)
        if record.label_path is None:
            reasons.append(f"LABEL_UNAVAILABLE:{record.sample_id}")
            continue
        label = (manifest_root / record.label_path).resolve()
        try:
            label.relative_to(manifest_root.resolve())
        except ValueError:
            reasons.append(f"LABEL_PATH_ESCAPE:{record.sample_id}")
            continue
        _, errors, counts = _validate_label(label, DEFECT_NAMES)
        if errors:
            reasons.extend(f"LABEL_INVALID:{record.sample_id}:{error}" for error in errors)
        split_boxes[split].update(counts)
    for group_id, splits in groups.items():
        if len(splits) > 1:
            reasons.append(f"GROUP_LEAKAGE:{group_id}")
    for split in _SPLITS:
        if split_images[split] == 0:
            reasons.append(f"EMPTY_SPLIT:{split}")
    total_boxes: Counter[str] = Counter()
    for counts in split_boxes.values():
        total_boxes.update(counts)
    represented = tuple(name for name in DEFECT_NAMES if total_boxes[name] > 0)
    minimum_images = 20 if stage == "B0" else 100
    minimum_classes = 2 if stage == "B0" else 3
    if len(accepted) < minimum_images:
        reasons.append(f"ACCEPTED_IMAGES_BELOW_{minimum_images}")
    if len(represented) < minimum_classes:
        reasons.append(f"REPRESENTED_CLASSES_BELOW_{minimum_classes}")
    if stage == "B1":
        for name in represented:
            if split_boxes["train"][name] < 30:
                reasons.append(f"TRAIN_BOX_COUNT_BELOW_30:{name}")
            if split_boxes["test"][name] < 10:
                reasons.append(f"TEST_BOX_COUNT_BELOW_10:{name}")
    return RevisionGate(
        stage=stage,
        status="blocked_data" if reasons else "ready",
        accepted_images=len(accepted),
        represented_classes=represented,
        train_boxes=dict(split_boxes["train"]),
        test_boxes=dict(split_boxes["test"]),
        split_images=dict(split_images),
        reasons=tuple(sorted(set(reasons))),
    )


def _canonical_manifest_hash(records: tuple[CandidateRecord, ...]) -> str:
    content = "".join(
        json.dumps(asdict(record), sort_keys=True, separators=(",", ":")) + "\n"
        for record in sorted(records, key=lambda item: item.sample_id)
    )
    return sha256(content.encode("utf-8")).hexdigest()


def _published_revision(root: Path, version: str) -> PublishedRevision:
    revision_json = root / "revision.json"
    document = json.loads(revision_json.read_text(encoding="utf-8"))
    return PublishedRevision(
        version=version,
        root=root,
        manifest=root / "manifest.jsonl",
        assignments=root / "assignments.jsonl",
        data_yaml=root / "data.yaml",
        revision_json=revision_json,
        manifest_sha256=str(document["published_manifest_sha256"]),
    )


def publish_revision(
    records: tuple[CandidateRecord, ...],
    manifest_root: Path,
    output_root: Path,
    *,
    version: str,
    stage: Stage,
    seed: int = 42,
) -> PublishedRevision:
    accepted = tuple(record for record in records if record.review_status == "accepted")
    accepted_hash = _canonical_manifest_hash(accepted)
    destination = output_root / version
    if destination.exists():
        revision_path = destination / "revision.json"
        try:
            existing = json.loads(revision_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("REVISION_IMMUTABLE") from exc
        if (
            existing.get("accepted_manifest_sha256") == accepted_hash
            and existing.get("stage") == stage
            and existing.get("split_seed") == seed
        ):
            return _published_revision(destination, version)
        raise ValueError("REVISION_IMMUTABLE")
    assignments = assign_group_stratified_v1(accepted, seed=seed)
    gate = evaluate_revision_gate(
        accepted,
        stage,
        manifest_root=manifest_root,
        assignments=assignments,
    )
    if gate.status != "ready":
        raise ValueError(f"REVISION_GATE_BLOCKED:{'|'.join(gate.reasons)}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{version}-", dir=output_root))
    try:
        manifest_lines: list[str] = []
        assignment_lines: list[str] = []
        for split in _SPLITS:
            (staging / split / "images").mkdir(parents=True)
            (staging / split / "labels").mkdir(parents=True)
        for record in sorted(accepted, key=lambda item: item.sample_id):
            split = assignments[record.sample_id]
            source_image = (manifest_root / record.image_path).resolve()
            source_label = (manifest_root / str(record.label_path)).resolve()
            image_relative = f"{split}/images/{record.sample_id}{source_image.suffix.lower()}"
            label_relative = f"{split}/labels/{record.sample_id}.txt"
            shutil.copy2(source_image, staging / image_relative)
            shutil.copy2(source_label, staging / label_relative)
            item = asdict(record)
            item.update({"split": split, "image_path": image_relative, "label_path": label_relative})
            manifest_lines.append(json.dumps(item, sort_keys=True) + "\n")
            assignment_lines.append(
                json.dumps(
                    {
                        "sample_id": record.sample_id,
                        "source_group_id": record.source_group_id,
                        "split": split,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
        manifest_path = staging / "manifest.jsonl"
        assignments_path = staging / "assignments.jsonl"
        manifest_path.write_text("".join(manifest_lines), encoding="utf-8")
        assignments_path.write_text("".join(assignment_lines), encoding="utf-8")
        data_document = {
            "path": staging.resolve().as_posix(),
            "train": "train/images",
            "val": "val/images",
            "test": "test/images",
            "names": {index: name for index, name in enumerate(DEFECT_NAMES)},
        }
        (staging / "data.yaml").write_text(
            yaml.safe_dump(data_document, sort_keys=False),
            encoding="utf-8",
        )
        revision = {
            "version": version,
            "stage": stage,
            "accepted_manifest_sha256": accepted_hash,
            "published_manifest_sha256": sha256_file(manifest_path),
            "assignments_sha256": sha256_file(assignments_path),
            "split_seed": seed,
            "split_algorithm": "group-stratified-v1",
            "accepted_images": gate.accepted_images,
            "represented_classes": list(gate.represented_classes),
            "train_boxes": dict(gate.train_boxes),
            "test_boxes": dict(gate.test_boxes),
            "split_images": dict(gate.split_images),
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        (staging / "revision.json").write_text(
            json.dumps(revision, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staging.replace(destination)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    published = _published_revision(destination, version)
    data_document["path"] = destination.resolve().as_posix()
    published.data_yaml.write_text(yaml.safe_dump(data_document, sort_keys=False), encoding="utf-8")
    return published


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish an immutable public FC-BGA dataset revision.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--stage", choices=("B0", "B1"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    sources = load_source_registry(args.sources)
    records = load_candidate_manifest(args.manifest, sources)
    revision = publish_revision(
        records,
        args.manifest.parent,
        args.output,
        version=args.version,
        stage=args.stage,
        seed=args.seed,
    )
    print(json.dumps({"version": revision.version, "root": str(revision.root)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
