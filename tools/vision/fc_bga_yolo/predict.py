from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

try:
    from .contracts import DEFECT_NAMES
    from .model_metadata import sha256_file, validate_model_package
except ImportError:
    from contracts import DEFECT_NAMES
    from model_metadata import sha256_file, validate_model_package


_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def resolve_prediction_inputs(source: Path) -> tuple[Path, ...]:
    if source.is_file() and source.suffix.lower() in _IMAGE_SUFFIXES:
        return (source.resolve(),)
    if source.is_dir():
        return tuple(sorted(path.resolve() for path in source.iterdir() if path.suffix.lower() in _IMAGE_SUFFIXES))
    if source.is_file() and source.suffix.lower() in {".jsonl", ".json"}:
        inputs: list[Path] = []
        for line in source.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            value = item.get("output_image")
            if not isinstance(value, str):
                raise ValueError("PREDICTION_MANIFEST_INVALID: output_image missing")
            inputs.append((source.parent / value).resolve())
        return tuple(inputs)
    raise ValueError("PREDICTION_SOURCE_INVALID")


def predict(
    model_path: Path,
    metadata_path: Path,
    source: Path,
    output_jsonl: Path,
    *,
    conf: float,
    device: str,
) -> Path:
    metadata = validate_model_package(model_path, metadata_path, DEFECT_NAMES)
    inputs = resolve_prediction_inputs(source)
    if not inputs:
        raise ValueError("PREDICTION_SOURCE_EMPTY")
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("install requirements-train.txt before prediction") from exc
    model = YOLO(str(model_path))
    results = model.predict(source=[str(path) for path in inputs], imgsz=metadata.imgsz, conf=conf, device=device, save=False, verbose=False)
    records: list[dict[str, object]] = []
    for path, result in zip(inputs, results, strict=True):
        detections: list[dict[str, object]] = []
        rows = result.boxes.data.tolist() if result.boxes is not None else []
        for x1, y1, x2, y2, confidence, class_id_value in rows:
            class_id = int(class_id_value)
            detections.append(
                {
                    "class_id": class_id,
                    "defect_code": DEFECT_NAMES[class_id],
                    "confidence": round(float(confidence), 8),
                    "x": int(round(x1)),
                    "y": int(round(y1)),
                    "w": int(round(x2 - x1)),
                    "h": int(round(y2 - y1)),
                }
            )
        records.append(
            {
                "image": str(path),
                "image_sha256": sha256_file(path),
                "model_sha256": metadata.model_sha256,
                "model_version": metadata.model_version,
                "detections": detections,
            }
        )
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_jsonl.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records), encoding="utf-8")
    return output_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description="Run deterministic FC-BGA YOLO prediction and write JSONL.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("predictions.jsonl"))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if not 0 <= args.conf <= 1:
        parser.error("--conf must be between 0 and 1")
    print(predict(args.model, args.metadata, args.source, args.output, conf=args.conf, device=args.device))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
