from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import shutil


MIN_WEIGHT_BYTES = 1024 * 1024


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


def download_model(model_name: str, destination: Path) -> WeightInfo:
    if Path(model_name).name != model_name or not model_name.endswith(".pt"):
        raise ValueError("model name must be a simple .pt filename")
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("install requirements-train.txt before downloading weights") from exc
    model = YOLO(model_name)
    source = Path(model.ckpt_path)
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / model_name
    if source.resolve() != target.resolve():
        shutil.copy2(source, target)
    return verify_weight(target)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download or verify official Ultralytics YOLOv8 weights.")
    parser.add_argument("--model", default="yolov8s.pt")
    parser.add_argument("--destination", type=Path, default=Path(__file__).parent / "weights")
    parser.add_argument("--verify-only", type=Path)
    args = parser.parse_args()
    info = verify_weight(args.verify_only) if args.verify_only else download_model(args.model, args.destination)
    print(json.dumps({**asdict(info), "path": str(info.path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
