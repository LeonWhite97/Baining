import os
from dataclasses import dataclass
from enum import StrEnum


class RuntimeMode(StrEnum):
    DEMO = "demo"
    SHADOW = "shadow"
    CONTROLLED = "controlled"


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
