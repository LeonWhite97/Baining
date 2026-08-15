from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from .contracts import INPUT_CONTRACT
except ImportError:
    from contracts import INPUT_CONTRACT


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_model_names(value: object) -> tuple[str, ...]:
    if isinstance(value, Mapping):
        integer_keys = set(range(len(value)))
        string_keys = {str(index) for index in range(len(value))}
        if set(value) == integer_keys:
            items = tuple(value[index] for index in range(len(value)))
        elif set(value) == string_keys:
            items = tuple(value[str(index)] for index in range(len(value)))
        else:
            raise ValueError("MODEL_CLASS_MISMATCH")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        items = tuple(value)
    else:
        raise ValueError("MODEL_CLASS_MISMATCH")
    if not items or any(not isinstance(item, str) or not item for item in items):
        raise ValueError("MODEL_CLASS_MISMATCH")
    return items


def validate_loaded_model_names(model: object, expected_names: tuple[str, ...]) -> None:
    try:
        actual_names = normalize_model_names(getattr(model, "names"))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("MODEL_CLASS_MISMATCH") from exc
    if actual_names != expected_names:
        raise ValueError("MODEL_CLASS_MISMATCH")


@dataclass(frozen=True, slots=True)
class ModelMetadata:
    model_version: str
    task: str
    names: tuple[str, ...]
    input_contract: str
    imgsz: int
    dataset_manifest_sha256: str
    model_sha256: str
    onnx_sha256: str | None
    runtime_versions: Mapping[str, str]
    export_settings: Mapping[str, object]
    result_paths: Mapping[str, str]
    intended_use: str

    @classmethod
    def from_dict(cls, item: Mapping[str, Any]) -> "ModelMetadata":
        required = {field.name for field in cls.__dataclass_fields__.values()}
        if set(item) != required:
            raise ValueError("MODEL_METADATA_INVALID: keys do not match the contract")
        try:
            return cls(
                model_version=str(item["model_version"]),
                task=str(item["task"]),
                names=tuple(item["names"]),
                input_contract=str(item["input_contract"]),
                imgsz=int(item["imgsz"]),
                dataset_manifest_sha256=str(item["dataset_manifest_sha256"]),
                model_sha256=str(item["model_sha256"]),
                onnx_sha256=str(item["onnx_sha256"]) if item["onnx_sha256"] is not None else None,
                runtime_versions=dict(item["runtime_versions"]),
                export_settings=dict(item["export_settings"]),
                result_paths=dict(item["result_paths"]),
                intended_use=str(item["intended_use"]),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("MODEL_METADATA_INVALID: field types are invalid") from exc


def write_model_metadata(path: Path, metadata: ModelMetadata) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(asdict(metadata), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def load_model_metadata(path: Path) -> ModelMetadata:
    try:
        item = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("MODEL_METADATA_INVALID: file cannot be read") from exc
    if not isinstance(item, dict):
        raise ValueError("MODEL_METADATA_INVALID: root must be an object")
    return ModelMetadata.from_dict(item)


def validate_model_package(
    model_path: Path,
    metadata_path: Path,
    expected_names: tuple[str, ...],
) -> ModelMetadata:
    metadata = load_model_metadata(metadata_path)
    if metadata.task != "detect" or metadata.intended_use != "portfolio_internal_poc":
        raise ValueError("MODEL_METADATA_INVALID: task or intended use is invalid")
    if metadata.names != expected_names:
        raise ValueError("MODEL_CLASS_MISMATCH")
    if metadata.input_contract != INPUT_CONTRACT:
        raise ValueError("MODEL_INPUT_CONTRACT_MISMATCH")
    if not model_path.is_file() or sha256_file(model_path) != metadata.model_sha256:
        raise ValueError("MODEL_HASH_MISMATCH")
    return metadata
