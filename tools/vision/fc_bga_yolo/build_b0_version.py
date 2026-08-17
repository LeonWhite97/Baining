from __future__ import annotations

"""Build (or pre-check) the B0 public-external revision from reviewed candidates.

This is a thin, friendly wrapper over the existing revision machinery in
`public_external_revision.py`:

  - `assign_group_stratified_v1`  -> 70/15/15 group-stratified train/val/test split
  - `evaluate_revision_gate`      -> B0 readiness (counts, classes, labels, leakage)
  - `publish_revision`            -> immutable materialization (images + labels +
                                     data.yaml + manifest + revision.json)

Why a wrapper: `publish_revision` raises a single opaque `REVISION_GATE_BLOCKED`
error. This script turns that into a readable checklist and only publishes when
the gate is actually ready, so the manual split step is fully automated the
moment the human annotation work lands.

Usage (from project root D:/YOLO/Baining):
  # dry-run: show the gate checklist + projected split (no writes)
  python tools/vision/fc_bga_yolo/build_b0_version.py

  # actually materialize versions/public-external-v0.1/ once the gate is ready
  python tools/vision/fc_bga_yolo/build_b0_version.py --publish
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from public_external_manifest import load_candidate_manifest, load_source_registry
from public_external_revision import (
    assign_group_stratified_v1,
    evaluate_revision_gate,
    publish_revision,
)

_REVIEW_MANIFEST = Path("data/external/fc_bga_public_external/review/candidates.jsonl")
_SOURCES = Path("data/external/fc_bga_public_external/sources.json")
_OUTPUT_ROOT = Path("data/external/fc_bga_public_external/versions")
_VERSION = "public-external-v0.1"
_STAGE = "B0"
_SEED = 42

# B0 thresholds (mirror evaluate_revision_gate)
_MIN_IMAGES = 20
_MIN_CLASSES = 2


def _classify_reasons(reasons: tuple[str, ...]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {
        "below_images": [],
        "below_classes": [],
        "empty_split": [],
        "label_missing": [],
        "label_invalid": [],
        "assignment_invalid": [],
        "group_leakage": [],
        "box_count": [],
    }
    for r in reasons:
        if r.startswith("ACCEPTED_IMAGES_BELOW"):
            out["below_images"].append(r)
        elif r.startswith("REPRESENTED_CLASSES_BELOW"):
            out["below_classes"].append(r)
        elif r.startswith("EMPTY_SPLIT"):
            out["empty_split"].append(r)
        elif r.startswith("LABEL_UNAVAILABLE"):
            out["label_missing"].append(r.split(":", 1)[1])
        elif r.startswith("LABEL_INVALID") or r.startswith("LABEL_PATH_ESCAPE"):
            out["label_invalid"].append(r)
        elif r.startswith("ASSIGNMENT_INVALID"):
            out["assignment_invalid"].append(r.split(":", 1)[1])
        elif r.startswith("GROUP_LEAKAGE"):
            out["group_leakage"].append(r.split(":", 1)[1])
        else:
            out["box_count"].append(r)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Build or pre-check the B0 public-external revision.")
    parser.add_argument("--manifest", type=Path, default=_REVIEW_MANIFEST)
    parser.add_argument("--sources", type=Path, default=_SOURCES)
    parser.add_argument("--output-root", type=Path, default=_OUTPUT_ROOT)
    parser.add_argument("--version", default=_VERSION)
    parser.add_argument("--stage", choices=("B0", "B1"), default=_STAGE)
    parser.add_argument("--seed", type=int, default=_SEED)
    parser.add_argument("--publish", action="store_true", help="Materialize the version once the gate is ready.")
    args = parser.parse_args()

    sources = load_source_registry(args.sources)
    records = load_candidate_manifest(args.manifest, sources)
    accepted = [r for r in records if r.review_status == "accepted"]

    print("=== B0 修订构建检查 ===")
    print(f"候选总数        : {len(records)}")
    print(f"已接受          : {len(accepted)}  (B0 需 ≥ {_MIN_IMAGES})")

    if not accepted:
        print()
        print("尚未有已接受样本。请先按 review/ANNOTATION_SPEC.md 完成人工标注：")
        print("  1. 在 candidates.jsonl 将样本 review_status 改为 accepted 并填 accepted_classes")
        print(f"  2. 至少接受 {_MIN_IMAGES} 张且覆盖 ≥ {_MIN_CLASSES} 个缺陷类")
        print("  3. 为已接受样本补 YOLO 边界框标签（label_path），再运行本脚本")
        return 1

    assignments = assign_group_stratified_v1(accepted, seed=args.seed)
    gate = evaluate_revision_gate(
        accepted, args.stage, manifest_root=args.manifest.parent, assignments=assignments
    )

    print(f"覆盖缺陷类      : {len(gate.represented_classes)} 个 -> {list(gate.represented_classes)}  (需 ≥ {_MIN_CLASSES})")
    print(f"分图计数        : {gate.split_images}")
    print(f"训练集框计数    : {gate.train_boxes}")
    print(f"测试集框计数    : {gate.test_boxes}")

    if gate.status == "ready":
        print()
        print("B0 门控：READY ✅")
        if args.publish:
            published = publish_revision(
                records,
                args.manifest.parent,
                args.output_root,
                version=args.version,
                stage=args.stage,
                seed=args.seed,
            )
            print(f"已发布修订      : {published.version}")
            print(f"修订根目录      : {published.root}")
            print(f"data.yaml       : {published.data_yaml}")
            print("B0 训练配置 tools/vision/fc_bga_yolo/configs/train_public_external_b0.yaml 现已指向有效数据集。")
        else:
            print("运行 `build_b0_version.py --publish` 以实体化 versions/public-external-v0.1/（不可变）。")
        return 0

    print()
    print("B0 门控：BLOCKED ❌")
    reasons = _classify_reasons(gate.reasons)
    if reasons["below_images"]:
        print(f"  - 已接受图不足：还差 {_MIN_IMAGES - len(accepted)} 张")
    if reasons["below_classes"]:
        print(f"  - 覆盖类别不足：还差 {_MIN_CLASSES - len(gate.represented_classes)} 个")
    if reasons["empty_split"]:
        print(f"  - 空子集：{reasons['empty_split']}")
    if reasons["label_missing"]:
        n = len(reasons["label_missing"])
        shown = ", ".join(reasons["label_missing"][:20])
        more = f" …(+{n - 20} 更多)" if n > 20 else ""
        print(f"  - {n} 个已接受样本缺边界框标签（label_path 未设置）：{shown}{more}")
        print("    需按 ANNOTATION_SPEC.md 补真实框标注，再写回 candidates.jsonl 的 label_path")
    if reasons["label_invalid"]:
        print(f"  - 标签无效/路径越界：{reasons['label_invalid'][:10]}")
    if reasons["group_leakage"]:
        print(f"  - 同组泄漏到多子集：{reasons['group_leakage'][:10]}")
    if reasons["assignment_invalid"]:
        print(f"  - 分配无效：{reasons['assignment_invalid'][:10]}")
    if reasons["box_count"]:
        print(f"  - 框数不足（B1 阈值）：{reasons['box_count'][:10]}")
    print()
    print("修复上述项后重跑本脚本；门控就绪后加 --publish 实体化。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
