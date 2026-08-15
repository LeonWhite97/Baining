import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class RuntimeMode(StrEnum):
    DEMO = "demo"
    SHADOW = "shadow"
    CONTROLLED = "controlled"


class InferenceBackend(StrEnum):
    DEMO = "demo"
    ULTRALYTICS = "ultralytics"
    TENSORRT = "tensorrt"


def _parse_bool(value: bool | str | None, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


@dataclass(frozen=True, slots=True)
class RuntimeSettings:
    mode: RuntimeMode
    auto_pass_enabled: bool
    handler_integration_enabled: bool

    @classmethod
    def from_values(
        cls,
        *,
        mode: str | RuntimeMode | None = None,
        auto_pass_enabled: bool | str | None = None,
        handler_integration_enabled: bool | str | None = None,
    ) -> "RuntimeSettings":
        raw_mode = mode or os.getenv("APP_MODE", RuntimeMode.DEMO.value)
        try:
            runtime_mode = RuntimeMode(raw_mode)
        except ValueError as exc:
            allowed = ", ".join(item.value for item in RuntimeMode)
            raise ValueError(f"Unsupported APP_MODE: {raw_mode}. Expected one of: {allowed}") from exc
        configured_auto_pass = auto_pass_enabled
        if configured_auto_pass is None:
            configured_auto_pass = os.getenv("AUTO_PASS_ENABLED")
        configured_handler = handler_integration_enabled
        if configured_handler is None:
            configured_handler = os.getenv("HANDLER_INTEGRATION_ENABLED")
        return cls(
            runtime_mode,
            _parse_bool(configured_auto_pass, default=False),
            _parse_bool(configured_handler, default=False),
        )


@dataclass(frozen=True, slots=True)
class InferenceSettings:
    backend: InferenceBackend | None
    model_path: Path | None
    metadata_path: Path | None
    device: str
    imgsz: int
    conf: float

    @classmethod
    def from_values(
        cls,
        *,
        backend: str | InferenceBackend | None = None,
        model_path: str | Path | None = None,
        metadata_path: str | Path | None = None,
        device: str | None = None,
        imgsz: str | int | None = None,
        conf: str | float | None = None,
    ) -> "InferenceSettings":
        raw_backend = backend if backend is not None else os.getenv("AOI_INFERENCE_BACKEND")
        selected_backend: InferenceBackend | None = None
        if raw_backend not in {None, ""}:
            try:
                selected_backend = InferenceBackend(raw_backend)
            except ValueError as exc:
                allowed = ", ".join(item.value for item in InferenceBackend)
                raise ValueError(f"Unsupported AOI_INFERENCE_BACKEND: {raw_backend}. Expected: {allowed}") from exc
        raw_model = model_path if model_path is not None else os.getenv("AOI_MODEL_PATH")
        raw_metadata = metadata_path if metadata_path is not None else os.getenv("AOI_MODEL_METADATA_PATH")
        raw_imgsz = imgsz if imgsz is not None else os.getenv("AOI_MODEL_IMGSZ", "1280")
        raw_conf = conf if conf is not None else os.getenv("AOI_MODEL_CONF", "0.25")
        try:
            parsed_imgsz = int(raw_imgsz)
        except (TypeError, ValueError) as exc:
            raise ValueError("AOI_MODEL_IMGSZ must be a positive integer") from exc
        try:
            parsed_conf = float(raw_conf)
        except (TypeError, ValueError) as exc:
            raise ValueError("AOI_MODEL_CONF must be between 0 and 1") from exc
        if parsed_imgsz <= 0:
            raise ValueError("AOI_MODEL_IMGSZ must be a positive integer")
        if not 0 <= parsed_conf <= 1:
            raise ValueError("AOI_MODEL_CONF must be between 0 and 1")
        return cls(
            backend=selected_backend,
            model_path=Path(raw_model) if raw_model else None,
            metadata_path=Path(raw_metadata) if raw_metadata else None,
            device=device or os.getenv("AOI_MODEL_DEVICE", "cpu"),
            imgsz=parsed_imgsz,
            conf=parsed_conf,
        )
