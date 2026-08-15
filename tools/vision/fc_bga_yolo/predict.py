from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Callable

from PIL import Image, ImageDraw

try:
    from .contracts import DEFECT_NAMES
    from .model_metadata import (
        sha256_file,
        validate_loaded_model_names,
        validate_model_package,
    )
except ImportError:
    from contracts import DEFECT_NAMES
    from model_metadata import sha256_file, validate_loaded_model_names, validate_model_package


_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _default_loader(path: Path) -> object:
    from ultralytics import YOLO

    return YOLO(str(path))


def _configuration_sha256(*, conf: float, device: str, imgsz: int, input_contract: str) -> str:
    payload = {
        "conf": conf,
        "device": device,
        "imgsz": imgsz,
        "input_contract": input_contract,
        "names": list(DEFECT_NAMES),
    }
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _annotate_image(source: Path, detections: list[dict[str, object]], output: Path) -> None:
    with Image.open(source) as opened:
        image = opened.convert("RGB")
    draw = ImageDraw.Draw(image)
    for detection in detections:
        x = int(detection["x"])
        y = int(detection["y"])
        w = int(detection["w"])
        h = int(detection["h"])
        label = f'{detection["defect_code"]} {float(detection["confidence"]):.3f}'
        draw.rectangle((x, y, x + w, y + h), outline=(220, 38, 38), width=2)
        draw.text((x, max(0, y - 11)), label, fill=(220, 38, 38))
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format="PNG")


def resolve_prediction_inputs(source: Path) -> tuple[Path, ...]:
    if source.is_file() and source.suffix.lower() in _IMAGE_SUFFIXES:
        return (source.resolve(),)
    if source.is_dir():
        return tuple(sorted(path.resolve() for path in source.iterdir() if path.suffix.lower() in _IMAGE_SUFFIXES))
    if source.is_file() and source.suffix.lower() in {".jsonl", ".json"}:
        inputs: list[Path] = []
        manifest_root = source.resolve().parent
        for line in source.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError("PREDICTION_MANIFEST_INVALID: record must be an object")
            value = item.get("output_image")
            if not isinstance(value, str):
                raise ValueError("PREDICTION_MANIFEST_INVALID: output_image missing")
            candidate = (manifest_root / value).resolve()
            try:
                candidate.relative_to(manifest_root)
            except ValueError as exc:
                raise ValueError("PREDICTION_PATH_OUTSIDE_ROOT") from exc
            if not candidate.is_file() or candidate.suffix.lower() not in _IMAGE_SUFFIXES:
                raise ValueError("PREDICTION_MANIFEST_INVALID: image unavailable")
            inputs.append(candidate)
        if len(inputs) != len(set(inputs)):
            raise ValueError("PREDICTION_MANIFEST_INVALID: duplicate image")
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
    annotated_dir: Path | None = None,
    summary_json: Path | None = None,
    model_loader: Callable[[Path], object] | None = None,
) -> Path:
    if not 0 <= conf <= 1:
        raise ValueError("PREDICTION_CONF_INVALID")
    metadata = validate_model_package(model_path, metadata_path, DEFECT_NAMES)
    inputs = resolve_prediction_inputs(source)
    if not inputs:
        raise ValueError("PREDICTION_SOURCE_EMPTY")
    try:
        model = (model_loader or _default_loader)(model_path)
    except ImportError as exc:
        raise RuntimeError("install requirements-train.txt before prediction") from exc
    validate_loaded_model_names(model, DEFECT_NAMES)
    configuration_sha256 = _configuration_sha256(
        conf=conf,
        device=device,
        imgsz=metadata.imgsz,
        input_contract=metadata.input_contract,
    )
    annotated_dir = annotated_dir or output_jsonl.parent / "annotated"
    summary_json = summary_json or output_jsonl.with_name(f"{output_jsonl.stem}.summary.json")
    results = model.predict(source=[str(path) for path in inputs], imgsz=metadata.imgsz, conf=conf, device=device, save=False, verbose=False)
    records: list[dict[str, object]] = []
    class_confidences: dict[str, list[float]] = {name: [] for name in DEFECT_NAMES}
    class_images: dict[str, int] = {name: 0 for name in DEFECT_NAMES}
    for index, (path, result) in enumerate(zip(inputs, results, strict=True), start=1):
        detections: list[dict[str, object]] = []
        detected_classes: set[str] = set()
        rows = result.boxes.data.tolist() if result.boxes is not None else []
        for x1, y1, x2, y2, confidence, class_id_value in rows:
            class_id = int(class_id_value)
            numeric = tuple(float(value) for value in (x1, y1, x2, y2, confidence))
            x1_value, y1_value, x2_value, y2_value, confidence_value = numeric
            if (
                float(class_id_value) != class_id
                or not 0 <= class_id < len(DEFECT_NAMES)
                or not all(math.isfinite(value) for value in numeric)
                or not 0 <= confidence_value <= 1
                or x2_value <= x1_value
                or y2_value <= y1_value
            ):
                raise ValueError("PREDICTION_ROW_INVALID")
            defect_code = DEFECT_NAMES[class_id]
            rounded_confidence = round(confidence_value, 8)
            detections.append(
                {
                    "class_id": class_id,
                    "defect_code": defect_code,
                    "confidence": rounded_confidence,
                    "x": int(round(x1_value)),
                    "y": int(round(y1_value)),
                    "w": int(round(x2_value - x1_value)),
                    "h": int(round(y2_value - y1_value)),
                }
            )
            class_confidences[defect_code].append(rounded_confidence)
            detected_classes.add(defect_code)
        for defect_code in detected_classes:
            class_images[defect_code] += 1
        image_sha256 = sha256_file(path)
        annotated_path = annotated_dir / f"{index:06d}_{image_sha256[:12]}.png"
        _annotate_image(path, detections, annotated_path)
        records.append(
            {
                "image": str(path),
                "image_sha256": image_sha256,
                "annotated_image": str(annotated_path),
                "model_sha256": metadata.model_sha256,
                "model_version": metadata.model_version,
                "configuration_sha256": configuration_sha256,
                "detections": detections,
            }
        )
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_jsonl.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records), encoding="utf-8")
    class_summary: dict[str, dict[str, object]] = {}
    for defect_code in DEFECT_NAMES:
        confidences = class_confidences[defect_code]
        class_summary[defect_code] = {
            "image_count": class_images[defect_code],
            "box_count": len(confidences),
            "confidence_min": min(confidences) if confidences else None,
            "confidence_max": max(confidences) if confidences else None,
            "confidence_mean": round(sum(confidences) / len(confidences), 8) if confidences else None,
        }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(
        json.dumps(
            {
                "images": len(records),
                "boxes": sum(len(record["detections"]) for record in records),
                "model_sha256": metadata.model_sha256,
                "model_version": metadata.model_version,
                "configuration_sha256": configuration_sha256,
                "classes": class_summary,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return output_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description="Run deterministic FC-BGA YOLO prediction and write JSONL.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("predictions.jsonl"))
    parser.add_argument("--annotated-dir", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if not 0 <= args.conf <= 1:
        parser.error("--conf must be between 0 and 1")
    print(
        predict(
            args.model,
            args.metadata,
            args.source,
            args.output,
            conf=args.conf,
            device=args.device,
            annotated_dir=args.annotated_dir,
            summary_json=args.summary,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
