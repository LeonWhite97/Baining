from __future__ import annotations

import argparse
from dataclasses import replace
import importlib.metadata
import platform
from pathlib import Path

try:
    from .contracts import DEFECT_NAMES
    from .model_metadata import sha256_file, validate_model_package, write_model_metadata
except ImportError:
    from contracts import DEFECT_NAMES
    from model_metadata import sha256_file, validate_model_package, write_model_metadata


def export_model(
    model_path: Path,
    metadata_path: Path,
    *,
    format_name: str,
    imgsz: int | None = None,
    device: str = "cpu",
) -> Path:
    if format_name not in {"onnx", "engine"}:
        raise ValueError("EXPORT_FORMAT_INVALID")
    metadata = validate_model_package(model_path, metadata_path, DEFECT_NAMES)
    try:
        import torch
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("install requirements-train.txt before export") from exc
    if format_name == "engine":
        if not torch.cuda.is_available():
            raise RuntimeError("TensorRT Engine export requires an available CUDA GPU")
        try:
            tensorrt_version = importlib.metadata.version("tensorrt")
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError("TensorRT Engine export requires the tensorrt package") from exc
    else:
        tensorrt_version = "not-applicable"
    model = YOLO(str(model_path))
    exported = Path(
        model.export(
            format=format_name,
            imgsz=imgsz or metadata.imgsz,
            device=device,
            dynamic=False,
            simplify=False,
        )
    )
    if not exported.is_file() or exported.stat().st_size < 1024:
        raise RuntimeError("export did not produce a valid model file")
    versions = dict(metadata.runtime_versions)
    versions.update(
        {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "ultralytics": importlib.metadata.version("ultralytics"),
            "cuda": str(torch.version.cuda),
            "tensorrt": tensorrt_version,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        }
    )
    updated = replace(
        metadata,
        onnx_sha256=sha256_file(exported) if format_name == "onnx" else metadata.onnx_sha256,
        runtime_versions=versions,
        export_settings={"format": format_name, "imgsz": imgsz or metadata.imgsz, "device": device},
    )
    write_model_metadata(metadata_path, updated)
    return exported


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a validated FC-BGA model package to ONNX or TensorRT Engine.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--format", choices=("onnx", "engine"), default="onnx")
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(export_model(args.model, args.metadata, format_name=args.format, imgsz=args.imgsz, device=args.device))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

