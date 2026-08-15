from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import shutil
from typing import Callable


MIN_WEIGHT_BYTES = 1024 * 1024
DEFAULT_MODELS = ("yolov8n.pt", "yolov8s.pt")


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
) -> tuple[WeightInfo, ...]:
    if not model_names or len(set(model_names)) != len(model_names):
        raise ValueError("models must be a non-empty unique list")
    infos: list[WeightInfo] = []
    for model_name in model_names:
        target = destination / model_name
        if target.is_file() and not force:
            infos.append(verify_weight(target))
        else:
            infos.append(downloader(model_name, destination, force=force))
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
        infos = prepare_models(names, args.destination, force=args.force)
    output = [{**asdict(info), "path": str(info.path)} for info in infos]
    print(json.dumps(output[0] if len(output) == 1 else output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
