from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import shutil
from typing import Callable

try:
    from .artifact_manifest import (
        ArtifactRecord,
        capture_artifact_record,
        load_artifact_records,
        verify_artifact_record,
        write_artifact_records,
    )
except ImportError:
    from artifact_manifest import (
        ArtifactRecord,
        capture_artifact_record,
        load_artifact_records,
        verify_artifact_record,
        write_artifact_records,
    )


MIN_WEIGHT_BYTES = 1024 * 1024
DEFAULT_MODELS = ("yolov8n.pt", "yolov8s.pt")
OFFICIAL_MODEL_URLS = {
    "yolov8n.pt": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt",
    "yolov8s.pt": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8s.pt",
}
ULTRALYTICS_LICENSE_URL = "https://www.ultralytics.com/license"


@dataclass(frozen=True, slots=True)
class WeightInfo:
    path: Path
    size_bytes: int
    sha256: str


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_weight(path: Path) -> WeightInfo:
    if not path.is_file():
        raise ValueError(f"weight is unavailable: {path.name}")
    size = path.stat().st_size
    if size < MIN_WEIGHT_BYTES:
        raise ValueError(f"weight is too small: {size} bytes")
    return WeightInfo(path=path.resolve(), size_bytes=size, sha256=sha256_file(path))


def download_model(model_name: str, destination: Path, *, force: bool = False) -> WeightInfo:
    if Path(model_name).name != model_name or not model_name.endswith(".pt"):
        raise ValueError("model name must be a simple .pt filename")
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / model_name
    if target.is_file() and not force:
        return verify_weight(target)
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("install requirements-train.txt before downloading weights") from exc
    model = YOLO(model_name)
    source = Path(model.ckpt_path)
    if source.resolve() != target.resolve():
        shutil.copy2(source, target)
    return verify_weight(target)


def prepare_models(
    model_names: tuple[str, ...],
    destination: Path,
    *,
    force: bool,
    downloader: Callable[..., WeightInfo] = download_model,
    manifest_path: Path | None = None,
    max_attempts: int = 3,
) -> tuple[WeightInfo, ...]:
    if not model_names or len(set(model_names)) != len(model_names):
        raise ValueError("models must be a non-empty unique list")
    if not 1 <= max_attempts <= 3:
        raise ValueError("OFFICIAL_DOWNLOAD_ATTEMPTS_INVALID")
    baselines: dict[str, ArtifactRecord] = {}
    if manifest_path is not None and manifest_path.is_file():
        baselines = {record.name: record for record in load_artifact_records(manifest_path)}
    infos: list[WeightInfo] = []
    for model_name in model_names:
        target = destination / model_name
        if target.is_file() and not force:
            info = verify_weight(target)
            if manifest_path is not None:
                baseline = baselines.get(model_name)
                if baseline is None:
                    raise ValueError("ARTIFACT_BASELINE_UNAVAILABLE")
                verify_artifact_record(target, baseline)
            infos.append(info)
        else:
            if manifest_path is not None and model_name not in OFFICIAL_MODEL_URLS:
                raise ValueError("OFFICIAL_MODEL_UNKNOWN")
            last_error: Exception | None = None
            for _ in range(max_attempts):
                try:
                    info = downloader(model_name, destination, force=force)
                    break
                except Exception as exc:
                    last_error = exc
            else:
                raise RuntimeError("OFFICIAL_DOWNLOAD_FAILED") from last_error
            infos.append(info)
            if manifest_path is not None:
                record = capture_artifact_record(
                    info.path,
                    source_url=OFFICIAL_MODEL_URLS[model_name],
                    license_url=ULTRALYTICS_LICENSE_URL,
                    retrieved_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                )
                baselines[model_name] = record
                write_artifact_records(manifest_path, tuple(baselines.values()))
    return tuple(infos)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download or verify official Ultralytics YOLOv8 weights.")
    parser.add_argument("--model", action="append", dest="model_aliases")
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--destination", type=Path, default=Path(__file__).parent / "weights/pretrained")
    parser.add_argument("--verify-only", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.models and args.model_aliases:
        parser.error("use either --models or --model, not both")
    if args.verify_only:
        infos = (verify_weight(args.verify_only),)
    else:
        names = tuple(args.models or args.model_aliases or DEFAULT_MODELS)
        infos = prepare_models(
            names,
            args.destination,
            force=args.force,
            manifest_path=args.destination / "artifact-manifest.json",
        )
    output = [{**asdict(info), "path": str(info.path)} for info in infos]
    print(json.dumps(output[0] if len(output) == 1 else output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
