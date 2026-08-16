from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable, Mapping

import numpy as np


EMPTY_CLASS_FOOTNOTE = "mAP is computed over classes with nonzero ground-truth instances only."
INSUFFICIENT_WARNING = (
    "INSUFFICIENT STATISTICAL EVIDENCE: workflow rehearsal metrics are not model-performance evidence."
)
ULTRALYTICS_EVALUATION_VERSION = "8.4.120"


@dataclass(frozen=True, slots=True)
class ImageValidationStats:
    sample_id: str
    source_group_id: str
    tp: np.ndarray
    conf: np.ndarray
    pred_cls: np.ndarray
    target_cls: np.ndarray


class ValidationStatsCollector:
    def __init__(self, group_by_filename: Mapping[str, tuple[str, str]]) -> None:
        normalized: dict[str, tuple[str, str]] = {}
        sample_ids: set[str] = set()
        for filename, identity in group_by_filename.items():
            basename = Path(filename).name
            sample_id, source_group_id = identity
            if not basename or basename in normalized:
                raise ValueError("VALIDATION_FILENAME_DUPLICATE")
            if not sample_id or sample_id in sample_ids or not source_group_id:
                raise ValueError("VALIDATION_IDENTITY_INVALID")
            normalized[basename] = (sample_id, source_group_id)
            sample_ids.add(sample_id)
        if not normalized:
            raise ValueError("VALIDATION_MAPPING_EMPTY")
        self._mapping = normalized
        self._expected_sample_ids = sample_ids
        self._cursor = 0
        self._records: list[ImageValidationStats] = []

    def on_val_batch_end(self, validator: object) -> None:
        metrics = getattr(validator, "metrics", None)
        stats = getattr(metrics, "stats", None)
        required = ("tp", "conf", "pred_cls", "target_cls")
        if not isinstance(stats, Mapping) or any(key not in stats for key in required):
            raise ValueError("VALIDATION_STATS_INVALID")
        if "im_name" in stats:
            image_names = tuple(str(value) for value in stats["im_name"])
        else:
            box = getattr(metrics, "box", None)
            image_metrics = getattr(box, "image_metrics", None)
            if not isinstance(image_metrics, Mapping):
                raise ValueError("VALIDATION_IMAGE_NAMES_INVALID")
            image_names = tuple(str(value) for value in image_metrics)
        lengths = {len(stats[key]) for key in required} | {len(image_names)}
        if len(lengths) != 1:
            raise ValueError("VALIDATION_STATS_LENGTH_MISMATCH")
        length = lengths.pop()
        if length < self._cursor:
            raise ValueError("VALIDATION_STATS_CURSOR_INVALID")
        existing = {record.sample_id for record in self._records}
        for index in range(self._cursor, length):
            basename = Path(image_names[index]).name
            identity = self._mapping.get(basename)
            if identity is None:
                raise ValueError(f"VALIDATION_IMAGE_UNMAPPED:{basename}")
            sample_id, source_group_id = identity
            if sample_id in existing:
                raise ValueError(f"VALIDATION_SAMPLE_DUPLICATE:{sample_id}")
            self._records.append(
                ImageValidationStats(
                    sample_id=sample_id,
                    source_group_id=source_group_id,
                    tp=np.asarray(stats["tp"][index]).copy(),
                    conf=np.asarray(stats["conf"][index]).copy(),
                    pred_cls=np.asarray(stats["pred_cls"][index]).copy(),
                    target_cls=np.asarray(stats["target_cls"][index]).copy(),
                )
            )
            existing.add(sample_id)
        self._cursor = length

    def records(self) -> tuple[ImageValidationStats, ...]:
        found = {record.sample_id for record in self._records}
        missing = self._expected_sample_ids - found
        if missing:
            raise ValueError(f"VALIDATION_SAMPLES_MISSING:{','.join(sorted(missing))}")
        return tuple(self._records)


def build_observed_class_report(
    *,
    names: tuple[str, ...],
    nt_per_class: np.ndarray,
    ap_class_index: np.ndarray,
    class_results: tuple[tuple[float, float, float, float], ...],
    native_results: tuple[float, float, float, float],
) -> dict[str, object]:
    counts = np.asarray(nt_per_class).reshape(-1)
    indexes = np.asarray(ap_class_index).reshape(-1)
    if len(counts) != len(names) or len(indexes) != len(class_results):
        raise ValueError("EVALUATION_CLASS_SHAPE_INVALID")
    rows = {int(class_id): tuple(float(value) for value in result) for class_id, result in zip(indexes, class_results)}
    classes: dict[str, object] = {}
    observed: list[tuple[float, float, float, float]] = []
    metric_names = ("precision", "recall", "mAP50", "mAP50_95")
    for class_id, name in enumerate(names):
        total_gt = int(counts[class_id])
        if total_gt == 0:
            classes[name] = {"total_gt": 0, "status": "no_evidence", "metrics": None}
            continue
        metrics = rows.get(class_id)
        if metrics is None:
            raise ValueError(f"EVALUATION_CLASS_RESULT_MISSING:{name}")
        observed.append(metrics)
        classes[name] = {
            "total_gt": total_gt,
            "status": "observed",
            "metrics": dict(zip(metric_names, metrics)),
        }
    if not observed:
        raise ValueError("EVALUATION_NO_OBSERVED_CLASSES")
    observed_array = np.asarray(observed, dtype=float)
    native = tuple(float(value) for value in native_results)
    if len(native) != 4:
        raise ValueError("EVALUATION_NATIVE_RESULT_INVALID")
    return {
        "warning": INSUFFICIENT_WARNING,
        "footnote": EMPTY_CLASS_FOOTNOTE,
        "observed_class_count": len(observed),
        "observed_class_mAP50": float(observed_array[:, 2].mean()),
        "observed_class_mAP50_95": float(observed_array[:, 3].mean()),
        "native_ultralytics": dict(zip(metric_names, native)),
        "classes": classes,
    }


def grouped_bootstrap_map(
    records: tuple[ImageValidationStats, ...],
    *,
    resamples: int = 1000,
    seed: int = 42,
    observer: Callable[[tuple[ImageValidationStats, ...]], None] | None = None,
) -> Mapping[str, tuple[float, float]]:
    if resamples < 1:
        raise ValueError("BOOTSTRAP_RESAMPLES_INVALID")
    groups: defaultdict[str, list[ImageValidationStats]] = defaultdict(list)
    for record in records:
        if not record.source_group_id:
            raise ValueError("BOOTSTRAP_SOURCE_GROUP_REQUIRED")
        groups[record.source_group_id].append(record)
    group_ids = tuple(sorted(groups))
    if len(group_ids) < 30:
        raise ValueError("BOOTSTRAP_GROUPS_BELOW_30")
    from ultralytics.utils.metrics import ap_per_class

    generator = np.random.default_rng(seed)
    map50_values: list[float] = []
    map_values: list[float] = []
    for _ in range(resamples):
        selected = generator.choice(group_ids, size=len(group_ids), replace=True)
        sampled = tuple(record for group_id in selected for record in groups[str(group_id)])
        if observer is not None:
            observer(sampled)
        tp = np.concatenate([record.tp for record in sampled], axis=0)
        conf = np.concatenate([record.conf for record in sampled])
        pred_cls = np.concatenate([record.pred_cls for record in sampled])
        target_cls = np.concatenate([record.target_cls for record in sampled])
        ap = np.asarray(ap_per_class(tp, conf, pred_cls, target_cls, plot=False)[5])
        if ap.size == 0:
            raise ValueError("BOOTSTRAP_NO_OBSERVED_CLASSES")
        map50_values.append(float(ap[:, 0].mean()))
        map_values.append(float(ap.mean()))
    return {
        "mAP50": tuple(float(value) for value in np.percentile(map50_values, [2.5, 97.5])),
        "mAP50_95": tuple(float(value) for value in np.percentile(map_values, [2.5, 97.5])),
    }


def write_public_evaluation_report(path: Path, report: Mapping[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path
