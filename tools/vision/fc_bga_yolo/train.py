from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import yaml

try:
    from .contracts import DEFECT_NAMES
    from .validate_yolo_dataset import validate_dataset
except ImportError:
    from contracts import DEFECT_NAMES
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
    project: str
    name: str


def load_training_settings(path: Path) -> TrainingSettings:
    try:
        item = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"TRAIN_CONFIG_INVALID: {path.name}") from exc
    if not isinstance(item, dict):
        raise ValueError("TRAIN_CONFIG_INVALID: root must be a mapping")
    required = {
        "profile", "model", "data", "imgsz", "epochs", "patience", "batch",
        "device", "workers", "seed", "conf", "project", "name",
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
            project=str(item["project"]),
            name=str(item["name"]),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("TRAIN_CONFIG_INVALID: field types are invalid") from exc
    if settings.profile not in {"fc_bga", "public_smoke"}:
        raise ValueError("TRAIN_CONFIG_INVALID: profile")
    if min(settings.imgsz, settings.epochs, settings.patience, settings.batch) < 1:
        raise ValueError("TRAIN_CONFIG_INVALID: positive numeric values required")
    if settings.workers < 0 or not 0 <= settings.conf <= 1:
        raise ValueError("TRAIN_CONFIG_INVALID: workers or conf")
    return settings


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
        "project": settings.project,
        "name": settings.name,
        "exist_ok": False,
    }


def _formal_dataset_root(data_yaml: Path) -> tuple[Path, Path | None]:
    document = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root_value = document.get("path") if isinstance(document, dict) else None
    if not isinstance(root_value, str):
        raise ValueError("DATA_CONFIG_INVALID: path")
    root = (data_yaml.parent / root_value).resolve()
    manifest = root / "manifest.jsonl"
    return root, manifest if manifest.is_file() else None


def check_training_settings(settings: TrainingSettings) -> None:
    data_yaml = Path(settings.data)
    if not data_yaml.is_file():
        raise ValueError(f"DATA_CONFIG_UNAVAILABLE: {data_yaml}")
    if settings.profile == "fc_bga":
        root, manifest = _formal_dataset_root(data_yaml)
        report = validate_dataset(root, DEFECT_NAMES, manifest)
        if report.errors or report.images == 0:
            detail = report.errors[0] if report.errors else "DATASET_EMPTY"
            raise ValueError(f"DATASET_INVALID: {detail}")


def run_training(settings: TrainingSettings) -> Path:
    check_training_settings(settings)
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("install requirements-train.txt before training") from exc
    model = YOLO(settings.model)
    results = model.train(**build_train_kwargs(settings))
    best = Path(results.save_dir) / "weights" / "best.pt"
    if not best.is_file():
        raise RuntimeError("training completed without weights/best.pt")
    model.val(data=settings.data, split="test", imgsz=settings.imgsz, device=settings.device, conf=settings.conf)
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description="Train or preflight an FC-BGA YOLOv8 detector.")
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "configs/train_poc.yaml")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    settings = load_training_settings(args.config)
    if args.check_only:
        check_training_settings(settings)
        print("training preflight passed")
        return 0
    best = run_training(settings)
    print(best)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

