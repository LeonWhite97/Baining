import pytest

from app.domain.capture import CaptureFrame, CaptureIdentity, CaptureValidationError, validate_capture


def identity(**overrides: object) -> CaptureIdentity:
    values: dict[str, object] = {
        "event_uuid": "EVENT-1",
        "cycle_id": "CYCLE-1",
        "capture_id": "f67af360-41a6-4a6c-9d15-7a9e6b28f194",
        "camera_trigger_sequence": 101,
        "previous_camera_trigger_sequence": 100,
        "start_received_monotonic_ns": 1_000,
        "capture_started_monotonic_ns": 1_100,
        "trigger_source": "HANDLER_START",
    }
    values.update(overrides)
    return CaptureIdentity(**values)


def frames(**overrides: object) -> list[CaptureFrame]:
    items = []
    for light_id in ("R", "G", "B", "RING"):
        values: dict[str, object] = {
            "capture_id": "f67af360-41a6-4a6c-9d15-7a9e6b28f194",
            "camera_trigger_sequence": 101,
            "light_id": light_id,
            "file_hash": light_id.lower() * 64,
        }
        values.update(overrides)
        items.append(CaptureFrame(**values))
    return items


def test_valid_capture_accepts_one_frame_per_required_light() -> None:
    validated = validate_capture(identity(), frames())

    assert validated.received_light_set == ("B", "G", "R", "RING")
    assert len(validated.input_fingerprint) == 64


@pytest.mark.parametrize(
    ("capture_overrides", "frame_overrides", "expected_code"),
    [
        ({"capture_started_monotonic_ns": 999}, {}, "CAPTURE_BEFORE_START"),
        ({"previous_camera_trigger_sequence": 101}, {}, "STALE_TRIGGER_SEQUENCE"),
        ({}, {"capture_id": "old-capture"}, "CAPTURE_ID_MISMATCH"),
        ({}, {"camera_trigger_sequence": 100}, "TRIGGER_SEQUENCE_MISMATCH"),
    ],
)
def test_capture_identity_rejects_stale_or_mismatched_frames(
    capture_overrides: dict[str, object],
    frame_overrides: dict[str, object],
    expected_code: str,
) -> None:
    with pytest.raises(CaptureValidationError) as exc_info:
        validate_capture(identity(**capture_overrides), frames(**frame_overrides))

    assert exc_info.value.code == expected_code
    assert exc_info.value.aoi_bin == 293


@pytest.mark.parametrize(
    ("captured_frames", "expected_code"),
    [
        (frames()[:-1], "MISSING_LIGHT"),
        (frames() + [frames()[0]], "DUPLICATE_LIGHT"),
        (frames() + [CaptureFrame(
            capture_id="f67af360-41a6-4a6c-9d15-7a9e6b28f194",
            camera_trigger_sequence=101,
            light_id="UV",
            file_hash="u" * 64,
        )], "UNKNOWN_LIGHT"),
    ],
)
def test_capture_rejects_incomplete_duplicate_or_unknown_lights(
    captured_frames: list[CaptureFrame], expected_code: str
) -> None:
    with pytest.raises(CaptureValidationError) as exc_info:
        validate_capture(identity(), captured_frames)

    assert exc_info.value.code == expected_code
    assert exc_info.value.aoi_bin == 293
