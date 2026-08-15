import pytest
from pydantic import ValidationError

from app.config import RuntimeMode, RuntimeSettings
from app.main import create_app
from app.schemas.handler import AOIStartRequest


def valid_start_payload() -> dict[str, str]:
    return {
        "trace_id": "TRACE-20260810-1",
        "handler_id": "HANDLER-1",
        "handler_session_id": "SESSION-1",
        "cycle_id": "CYCLE-1",
        "station_code": "AOI",
        "product_id": "BGA-256",
        "batch_id": "LOT-1",
        "tray_id": "TRAY-1",
        "slot_index": "01",
        "surface": "TOP",
    }


def test_start_schema_rejects_scenario_field() -> None:
    payload = valid_start_payload()
    payload["scenario"] = "NORMAL"

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        AOIStartRequest.model_validate(payload)


def test_unknown_runtime_mode_fails_app_creation() -> None:
    with pytest.raises(ValueError, match="Unsupported APP_MODE"):
        create_app(database_url="sqlite+pysqlite:///:memory:", mode="production")


def test_auto_pass_defaults_disabled() -> None:
    settings = RuntimeSettings.from_values(mode="shadow", auto_pass_enabled=None)

    assert settings.mode is RuntimeMode.SHADOW
    assert settings.auto_pass_enabled is False
    assert settings.handler_integration_enabled is False
