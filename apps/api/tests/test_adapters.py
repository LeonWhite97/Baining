from pathlib import Path

import pytest

from app.adapters.pis_in import IdentityUnavailable, PisInSourceAdapter
from app.inference.base import InferenceRequest, InferenceUnavailable
from app.inference.demo import DemoInferenceAdapter
from app.inference.tensorrt import TensorRtInferenceAdapter


def test_pis_in_adapter_normalizes_lights_without_splitting_event() -> None:
    raw = {
        "DeviceID": "PIS-01",
        "DeviceSessionID": "BOOT-77",
        "InspectionSequence": "1042",
        "ProductID": "BGA-256",
        "BatchID": "LOT-09",
        "TrayID": "TRAY-09",
        "SlotIndex": "A07",
        "Station": "ST-01",
        "Surface": "TOP",
        "Images": [
            {"LightGroup": light, "Path": f"/drop/{light}.png", "SHA256": light * 64}
            for light in ("R", "G", "B", "RING")
        ],
    }

    normalized = PisInSourceAdapter().normalize(raw)

    assert normalized.device_session_id == "BOOT-77"
    assert normalized.slot_index == "A07"
    assert [item.light_id for item in normalized.attachments] == ["R", "G", "B", "RING"]
    assert "light" not in normalized.source_key_hash.lower()


def test_pis_in_adapter_quarantines_missing_identity() -> None:
    with pytest.raises(IdentityUnavailable, match="DeviceSessionID"):
        PisInSourceAdapter().normalize({"DeviceID": "PIS-01"})


def test_demo_inference_is_deterministic() -> None:
    request = InferenceRequest(event_uuid="event-42", scenario="DEFECT", input_complete=True)

    first = DemoInferenceAdapter().predict(request)
    second = DemoInferenceAdapter().predict(request)

    assert first == second
    assert first.defect_code == "BALL_BRIDGE"
    assert first.defect_score >= 0.75


def test_missing_tensorrt_engine_fails_closed() -> None:
    adapter = TensorRtInferenceAdapter(Path("tests/fixtures/definitely-missing.engine"))

    with pytest.raises(InferenceUnavailable, match="engine"):
        adapter.predict(InferenceRequest(event_uuid="event-1", scenario="NORMAL", input_complete=True))
