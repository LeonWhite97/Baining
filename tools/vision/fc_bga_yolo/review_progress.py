from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from public_external_manifest import (
    CandidateRecord,
    load_candidate_manifest,
    load_source_registry,
)
from public_external_revision import evaluate_revision_gate

# B0 gate thresholds copied from public_external_revision.evaluate_revision_gate
_B0_MIN_IMAGES = 20
_B0_MIN_CLASSES = 2


@dataclass(frozen=True, slots=True)
class ReviewProgress:
    total: int
    by_status: dict[str, int]
    by_source: dict[str, int]
    accepted: int
    accepted_classes: tuple[str, ...]
    represented_class_count: int
    b0_status: str
    b0_reasons: tuple[str, ...]
    images_to_b0: int
    classes_to_b0: int

    def as_dict(self) -> dict:
        return asdict(self)


def summarize_review(manifest: Path, sources: Path) -> ReviewProgress:
    source_records = load_source_registry(sources)
    candidates = load_candidate_manifest(manifest, source_records)

    by_status: Counter[str] = Counter(c.review_status for c in candidates)
    by_source: Counter[str] = Counter(c.source_id for c in candidates)
    accepted = tuple(c for c in candidates if c.review_status == "accepted")

    represented: set[str] = set()
    for c in accepted:
        represented.update(c.accepted_classes)

    gate = evaluate_revision_gate(accepted, "B0", manifest_root=manifest.parent)

    images_to_b0 = max(_B0_MIN_IMAGES - len(accepted), 0)
    classes_to_b0 = max(_B0_MIN_CLASSES - len(represented), 0)

    return ReviewProgress(
        total=len(candidates),
        by_status=dict(by_status),
        by_source=dict(by_source),
        accepted=len(accepted),
        accepted_classes=tuple(sorted(represented)),
        represented_class_count=len(represented),
        b0_status=gate.status,
        b0_reasons=tuple(sorted(set(gate.reasons))),
        images_to_b0=images_to_b0,
        classes_to_b0=classes_to_b0,
    )


def _print_progress(progress: ReviewProgress) -> None:
    print("=== FC-BGA 公开外部候选审查进度 ===")
    print(f"总候选数        : {progress.total}")
    print(f"按状态          : {progress.by_status}")
    print(f"按数据源        : {progress.by_source}")
    print(f"已接受          : {progress.accepted}")
    print(f"已覆盖七类      : {progress.represented_class_count} 个 -> {list(progress.accepted_classes)}")
    print()
    print(f"--- B0 门控 ({_B0_MIN_IMAGES} 图 / {_B0_MIN_CLASSES} 类) ---")
    print(f"状态            : {progress.b0_status}")
    if progress.b0_status == "blocked_data":
        print(f"距解锁还差      : {progress.images_to_b0} 张已接受图, {progress.classes_to_b0} 个类别")
        print(f"阻塞原因        : {list(progress.b0_reasons)}")
    else:
        print("已满足 B0 门控，可发布 public-external-v0.1")


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize public FC-BGA candidate review progress.")
    parser.add_argument("--manifest", type=Path, default=Path("data/external/fc_bga_public_external/review/candidates.jsonl"))
    parser.add_argument("--sources", type=Path, default=Path("data/external/fc_bga_public_external/sources.json"))
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON instead of text.")
    args = parser.parse_args()

    progress = summarize_review(args.manifest, args.sources)
    if args.json:
        print(json.dumps(progress.as_dict(), indent=2, ensure_ascii=False))
    else:
        _print_progress(progress)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
