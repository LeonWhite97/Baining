from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import importlib.metadata
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


@dataclass(frozen=True, slots=True)
class TrainingArtifacts:
    save_dir: Path
    best: Path
    last: Path


def _validate_training_settings(settings: TrainingSettings) -> TrainingSettings:
    if settings.profile not in {"fc_bga", "public_smoke"}:
        raise ValueError("TRAIN_CONFIG_INVALID: profile")
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
    if set(item) != required:
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


def build_training_metadata(
    settings: TrainingSettings,
    model_path: Path,
    *,
    result_paths: Mapping[str, str],
    runtime_versions: Mapping[str, str],
) -> ModelMetadata:
    if settings.profile != "fc_bga":
        raise ValueError("PUBLIC_SMOKE_MODEL_NOT_DEPLOYABLE")
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
    kwargs = build_train_kwargs(settings)
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
    if settings.profile == "fc_bga":
        expected_names = DEFECT_NAMES
    else:
        expected_names = normalize_model_names(_data_document(Path(settings.data)).get("names"))
    validate_loaded_model_names(best_model, expected_names)
    validation = best_model.val(
        data=settings.data,
        split="test",
        imgsz=settings.imgsz,
        device=settings.device,
        conf=settings.conf,
    )
    result_paths = {
        "train": str(artifacts.save_dir),
        "test": str(Path(validation.save_dir)),
    }
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
