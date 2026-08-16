from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import importlib.metadata
import json
import platform
from pathlib import Path
from typing import Mapping

import yaml

try:
    from .contracts import DEFECT_NAMES, INPUT_CONTRACT
    from .model_metadata import (
        ModelMetadata,
        normalize_model_names,
        sha256_file,
        validate_loaded_model_names,
        write_model_metadata,
    )
    from .public_external_evaluation import (
        ValidationStatsCollector,
        build_observed_class_report,
        grouped_bootstrap_map,
        write_public_evaluation_report,
    )
    from .validate_yolo_dataset import validate_dataset
except ImportError:
    from contracts import DEFECT_NAMES, INPUT_CONTRACT
    from model_metadata import (
        ModelMetadata,
        normalize_model_names,
        sha256_file,
        validate_loaded_model_names,
        write_model_metadata,
    )
    from public_external_evaluation import (
        ValidationStatsCollector,
        build_observed_class_report,
        grouped_bootstrap_map,
        write_public_evaluation_report,
    )
    from validate_yolo_dataset import validate_dataset


@dataclass(frozen=True, slots=True)
class TrainingSettings:
    profile: str
    model: str
    data: str
    imgsz: int
    epochs: int
    patience: int
    batch: int
    device: str
    workers: int
    seed: int
    conf: float
    lr0: float
    project: str
    name: str
    public_stage: str | None = None
    dataset_revision: str | None = None


@dataclass(frozen=True, slots=True)
class TrainingArtifacts:
    save_dir: Path
    best: Path
    last: Path


def _validate_training_settings(settings: TrainingSettings) -> TrainingSettings:
    if settings.profile not in {"fc_bga", "public_smoke", "public_external"}:
        raise ValueError("TRAIN_CONFIG_INVALID: profile")
    if settings.profile == "public_external":
        expected_revisions = {"B0": "public-external-v0.1", "B1": "public-external-v0.2"}
        if expected_revisions.get(settings.public_stage) != settings.dataset_revision:
            raise ValueError("TRAIN_CONFIG_INVALID: public revision")
    elif settings.public_stage is not None or settings.dataset_revision is not None:
        raise ValueError("TRAIN_CONFIG_INVALID: unexpected public revision")
    if min(settings.imgsz, settings.epochs, settings.patience, settings.batch) < 1:
        raise ValueError("TRAIN_CONFIG_INVALID: positive numeric values required")
    if settings.workers < 0 or not 0 <= settings.conf <= 1 or settings.lr0 <= 0:
        raise ValueError("TRAIN_CONFIG_INVALID: workers, conf, or lr0")
    return settings


def load_training_settings(path: Path) -> TrainingSettings:
    try:
        item = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"TRAIN_CONFIG_INVALID: {path.name}") from exc
    if not isinstance(item, dict):
        raise ValueError("TRAIN_CONFIG_INVALID: root must be a mapping")
    required = {
        "profile", "model", "data", "imgsz", "epochs", "patience", "batch",
        "device", "workers", "seed", "conf", "lr0", "project", "name",
    }
    public_fields = {"public_stage", "dataset_revision"}
    if not required.issubset(item) or set(item) - required - public_fields:
        raise ValueError("TRAIN_CONFIG_INVALID: keys do not match the contract")
    try:
        settings = TrainingSettings(
            profile=str(item["profile"]),
            model=str(item["model"]),
            data=str(item["data"]),
            imgsz=int(item["imgsz"]),
            epochs=int(item["epochs"]),
            patience=int(item["patience"]),
            batch=int(item["batch"]),
            device=str(item["device"]),
            workers=int(item["workers"]),
            seed=int(item["seed"]),
            conf=float(item["conf"]),
            lr0=float(item["lr0"]),
            project=str(item["project"]),
            name=str(item["name"]),
            public_stage=(str(item["public_stage"]) if "public_stage" in item else None),
            dataset_revision=(
                str(item["dataset_revision"]) if "dataset_revision" in item else None
            ),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("TRAIN_CONFIG_INVALID: field types are invalid") from exc
    return _validate_training_settings(settings)


def apply_training_overrides(
    settings: TrainingSettings,
    *,
    model: str | None = None,
    data: str | None = None,
    imgsz: int | None = None,
    epochs: int | None = None,
    patience: int | None = None,
    batch: int | None = None,
    device: str | None = None,
    workers: int | None = None,
    seed: int | None = None,
    conf: float | None = None,
    lr0: float | None = None,
    project: str | None = None,
    name: str | None = None,
) -> TrainingSettings:
    values = {
        "model": model,
        "data": data,
        "imgsz": imgsz,
        "epochs": epochs,
        "patience": patience,
        "batch": batch,
        "device": device,
        "workers": workers,
        "seed": seed,
        "conf": conf,
        "lr0": lr0,
        "project": project,
        "name": name,
    }
    overridden = replace(
        settings,
        **{key: value for key, value in values.items() if value is not None},
    )
    return _validate_training_settings(overridden)


def build_train_kwargs(settings: TrainingSettings) -> dict[str, object]:
    return {
        "data": settings.data,
        "imgsz": settings.imgsz,
        "epochs": settings.epochs,
        "patience": settings.patience,
        "batch": settings.batch,
        "device": settings.device,
        "workers": settings.workers,
        "seed": settings.seed,
        "deterministic": True,
        "lr0": settings.lr0,
        "project": settings.project,
        "name": settings.name,
        "exist_ok": False,
    }


def _data_document(data_yaml: Path) -> dict[str, object]:
    try:
        document = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"DATA_CONFIG_INVALID: {data_yaml.name}") from exc
    if not isinstance(document, dict):
        raise ValueError("DATA_CONFIG_INVALID: root must be a mapping")
    return document


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_dataset_root(data_yaml: Path, root_value: str) -> Path:
    root_path = Path(root_value)
    if root_path.is_absolute():
        return root_path.resolve()
    candidates = (
        (data_yaml.parent / root_path).resolve(),
        (Path.cwd() / root_path).resolve(),
        (_repo_root() / root_path).resolve(),
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _materialize_training_data(settings: TrainingSettings) -> Path:
    source = Path(settings.data)
    document = _data_document(source)
    root_value = document.get("path")
    if isinstance(root_value, str):
        document["path"] = _resolve_dataset_root(source, root_value).as_posix()
    runtime_dir = Path(settings.project) / ".resolved-data"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    destination = runtime_dir / f"{settings.name}.yaml"
    temporary = destination.with_name(f"{destination.name}.tmp")
    temporary.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    temporary.replace(destination)
    return destination.resolve()


def _formal_dataset_root(data_yaml: Path) -> tuple[Path, Path]:
    document = _data_document(data_yaml)
    root_value = document.get("path") if isinstance(document, dict) else None
    if not isinstance(root_value, str):
        raise ValueError("DATA_CONFIG_INVALID: path")
    for split in ("train", "val", "test"):
        if not isinstance(document.get(split), str) or not document[split]:
            raise ValueError(f"DATA_CONFIG_INVALID: {split}")
        if document[split] != f"{split}/images":
            raise ValueError(f"DATA_SPLIT_PATH_MISMATCH:{split}")
    try:
        names = normalize_model_names(document.get("names"))
    except ValueError as exc:
        raise ValueError("DATA_CLASS_MISMATCH") from exc
    if names != DEFECT_NAMES:
        raise ValueError("DATA_CLASS_MISMATCH")
    root = _resolve_dataset_root(data_yaml, root_value)
    manifest = root / "manifest.jsonl"
    if not manifest.is_file():
        raise ValueError("DATASET_INVALID: MANIFEST_UNAVAILABLE")
    return root, manifest


def _public_external_dataset_root(
    data_yaml: Path,
    settings: TrainingSettings,
) -> Path:
    document = _data_document(data_yaml)
    root_value = document.get("path")
    if not isinstance(root_value, str):
        raise ValueError("DATA_CONFIG_INVALID: path")
    for split in ("train", "val", "test"):
        if document.get(split) != f"{split}/images":
            raise ValueError(f"DATA_SPLIT_PATH_MISMATCH:{split}")
    try:
        names = normalize_model_names(document.get("names"))
    except ValueError as exc:
        raise ValueError("DATA_CLASS_MISMATCH") from exc
    if names != DEFECT_NAMES:
        raise ValueError("DATA_CLASS_MISMATCH")
    root = _resolve_dataset_root(data_yaml, root_value)
    revision_path = root / "revision.json"
    manifest = root / "manifest.jsonl"
    assignments = root / "assignments.jsonl"
    try:
        revision = json.loads(revision_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError) as exc:
        raise ValueError("REVISION_METADATA_INVALID") from exc
    if not isinstance(revision, dict):
        raise ValueError("REVISION_METADATA_INVALID")
    if revision.get("stage") != settings.public_stage:
        raise ValueError("REVISION_STAGE_MISMATCH")
    if revision.get("version") != settings.dataset_revision:
        raise ValueError("REVISION_VERSION_MISMATCH")
    expected_manifest_hash = revision.get("published_manifest_sha256")
    if (
        not isinstance(expected_manifest_hash, str)
        or not manifest.is_file()
        or sha256_file(manifest) != expected_manifest_hash
    ):
        raise ValueError("REVISION_MANIFEST_HASH_MISMATCH")
    expected_assignments_hash = revision.get("assignments_sha256")
    if (
        not isinstance(expected_assignments_hash, str)
        or not assignments.is_file()
        or sha256_file(assignments) != expected_assignments_hash
    ):
        raise ValueError("REVISION_ASSIGNMENTS_HASH_MISMATCH")
    image_suffixes = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
    for split in ("train", "val", "test"):
        image_dir = root / split / "images"
        if not image_dir.is_dir() or not any(
            path.is_file() and path.suffix.lower() in image_suffixes
            for path in image_dir.iterdir()
        ):
            raise ValueError(f"DATASET_INVALID: EMPTY_SPLIT:{split}")
    return root


def _public_test_group_mapping(root: Path) -> Mapping[str, tuple[str, str]]:
    mapping: dict[str, tuple[str, str]] = {}
    try:
        lines = (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ValueError("REVISION_MANIFEST_INVALID") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"REVISION_MANIFEST_INVALID:{line_number}") from exc
        if not isinstance(item, dict) or item.get("split") != "test":
            continue
        sample_id = item.get("sample_id")
        group_id = item.get("source_group_id")
        image_path = item.get("image_path")
        if not all(isinstance(value, str) and value for value in (sample_id, group_id, image_path)):
            raise ValueError(f"REVISION_MANIFEST_INVALID:{line_number}")
        mapping[Path(image_path).name] = (sample_id, group_id)
    if not mapping:
        raise ValueError("DATASET_INVALID: EMPTY_SPLIT:test")
    return mapping


def check_training_settings(settings: TrainingSettings) -> None:
    data_yaml = Path(settings.data)
    if not data_yaml.is_file():
        raise ValueError(f"DATA_CONFIG_UNAVAILABLE: {data_yaml}")
    if settings.profile == "fc_bga":
        root, manifest = _formal_dataset_root(data_yaml)
        report = validate_dataset(
            root,
            DEFECT_NAMES,
            manifest,
            require_nonempty_splits=True,
        )
        for split in ("train", "val", "test"):
            if report.split_images.get(split, 0) == 0:
                raise ValueError(f"DATASET_INVALID: EMPTY_SPLIT:{split}")
        if report.errors or report.images == 0:
            detail = report.errors[0] if report.errors else "DATASET_EMPTY"
            raise ValueError(f"DATASET_INVALID: {detail}")
    elif settings.profile == "public_external":
        _public_external_dataset_root(data_yaml, settings)


def build_training_metadata(
    settings: TrainingSettings,
    model_path: Path,
    *,
    result_paths: Mapping[str, str],
    runtime_versions: Mapping[str, str],
) -> ModelMetadata:
    if settings.profile == "public_smoke":
        raise ValueError("PUBLIC_SMOKE_MODEL_NOT_DEPLOYABLE")
    if settings.profile == "public_external":
        raise ValueError("PUBLIC_EXTERNAL_MODEL_NOT_DEPLOYABLE")
    data_yaml = Path(settings.data)
    dataset_artifact = data_yaml
    if data_yaml.is_file():
        root, manifest = _formal_dataset_root(data_yaml)
        dataset_artifact = manifest
    if not dataset_artifact.is_file():
        raise ValueError(f"DATA_CONFIG_UNAVAILABLE: {dataset_artifact}")
    return ModelMetadata(
        model_version=settings.name,
        task="detect",
        names=DEFECT_NAMES,
        input_contract=INPUT_CONTRACT,
        imgsz=settings.imgsz,
        dataset_manifest_sha256=sha256_file(dataset_artifact),
        model_sha256=sha256_file(model_path),
        onnx_sha256=None,
        runtime_versions=dict(runtime_versions),
        export_settings={},
        result_paths=dict(result_paths),
        intended_use="portfolio_internal_poc",
    )


def train_only(
    settings: TrainingSettings,
    *,
    epochs: int | None = None,
    resume_from: Path | None = None,
) -> TrainingArtifacts:
    check_training_settings(settings)
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("install requirements-train.txt before training") from exc
    if epochs is not None and epochs < 1:
        raise ValueError("TRAIN_EPOCHS_INVALID")
    if resume_from is not None and not resume_from.is_file():
        raise ValueError("TRAIN_RESUME_CHECKPOINT_UNAVAILABLE")
    model = YOLO(str(resume_from) if resume_from is not None else settings.model)
    runtime_settings = replace(settings, data=str(_materialize_training_data(settings)))
    kwargs = build_train_kwargs(runtime_settings)
    if epochs is not None:
        kwargs["epochs"] = epochs
    if resume_from is not None:
        kwargs["resume"] = str(resume_from)
    results = model.train(**kwargs)
    save_dir = Path(results.save_dir)
    best = save_dir / "weights" / "best.pt"
    last = save_dir / "weights" / "last.pt"
    if not best.is_file() or not last.is_file():
        raise RuntimeError("training completed without weights/best.pt and weights/last.pt")
    return TrainingArtifacts(save_dir=save_dir, best=best, last=last)


def evaluate_best(settings: TrainingSettings, artifacts: TrainingArtifacts) -> object:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("install requirements-train.txt before training") from exc
    best_model = YOLO(str(artifacts.best))
    collector = None
    if settings.profile == "fc_bga":
        expected_names = DEFECT_NAMES
    elif settings.profile == "public_external":
        root = _public_external_dataset_root(Path(settings.data), settings)
        collector = ValidationStatsCollector(_public_test_group_mapping(root))
        best_model.add_callback("on_val_batch_end", collector.on_val_batch_end)
        expected_names = DEFECT_NAMES
    else:
        expected_names = normalize_model_names(_data_document(Path(settings.data)).get("names"))
    validate_loaded_model_names(best_model, expected_names)
    validation = best_model.val(
        data=str(_materialize_training_data(settings)),
        split="test",
        imgsz=settings.imgsz,
        device=settings.device,
        conf=settings.conf,
    )
    result_paths = {
        "train": str(artifacts.save_dir),
        "test": str(Path(validation.save_dir)),
    }
    if settings.profile == "public_external":
        assert collector is not None
        records = collector.records()
        class_indexes = tuple(int(value) for value in validation.ap_class_index)
        report = build_observed_class_report(
            names=DEFECT_NAMES,
            nt_per_class=validation.nt_per_class,
            ap_class_index=validation.ap_class_index,
            class_results=tuple(
                tuple(float(value) for value in validation.class_result(index))
                for index in range(len(class_indexes))
            ),
            native_results=tuple(float(value) for value in validation.mean_results()),
        )
        report["test_images"] = len(records)
        report["test_boxes"] = int(sum(int(value) for value in validation.nt_per_class))
        report["test_source_groups"] = len({record.source_group_id for record in records})
        if settings.public_stage == "B1" and report["test_source_groups"] >= 30:
            report["bootstrap_95"] = dict(grouped_bootstrap_map(records, resamples=1000, seed=42))
        write_public_evaluation_report(
            Path(validation.save_dir) / "public_evaluation_report.json",
            report,
        )
    if settings.profile == "fc_bga":
        metadata = build_training_metadata(
            settings,
            artifacts.best,
            result_paths=result_paths,
            runtime_versions={
                "python": platform.python_version(),
                "pytorch": importlib.metadata.version("torch"),
                "ultralytics": importlib.metadata.version("ultralytics"),
            },
        )
        write_model_metadata(artifacts.best.parent / "model_metadata.json", metadata)
    return validation


def run_training(settings: TrainingSettings) -> Path:
    artifacts = train_only(settings)
    evaluate_best(settings, artifacts)
    return artifacts.best


def main() -> int:
    parser = argparse.ArgumentParser(description="Train or preflight an FC-BGA YOLOv8 detector.")
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "configs/train_poc.yaml")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--data")
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--device")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--conf", type=float)
    parser.add_argument("--lr0", type=float)
    parser.add_argument("--project")
    parser.add_argument("--name")
    args = parser.parse_args()
    settings = apply_training_overrides(
        load_training_settings(args.config),
        model=args.model,
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        patience=args.patience,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        conf=args.conf,
        lr0=args.lr0,
        project=args.project,
        name=args.name,
    )
    if args.check_only:
        check_training_settings(settings)
        print("training preflight passed")
        return 0
    best = run_training(settings)
    print(best)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
