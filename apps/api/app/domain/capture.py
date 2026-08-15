import hashlib
import json
from dataclasses import dataclass
from typing import Literal


RequiredLight = Literal["R", "G", "B", "RING"]
TriggerSource = Literal["HANDLER_START", "MANUAL", "RETRY"]
REQUIRED_LIGHTS = frozenset({"R", "G", "B", "RING"})


class CaptureValidationError(ValueError):
    def __init__(self, code: str, message: str, *, aoi_bin: int = 293) -> None:
        super().__init__(message)
        self.code = code
        self.aoi_bin = aoi_bin


@dataclass(frozen=True, slots=True)
class CaptureIdentity:
    event_uuid: str
    cycle_id: str
    capture_id: str
    camera_trigger_sequence: int
    previous_camera_trigger_sequence: int | None
    start_received_monotonic_ns: int
    capture_started_monotonic_ns: int
    trigger_source: TriggerSource


@dataclass(frozen=True, slots=True)
class CaptureFrame:
    capture_id: str
    camera_trigger_sequence: int
    light_id: str
    file_hash: str
    data_type: str = "2D_IMAGE"


@dataclass(frozen=True, slots=True)
class ValidatedCapture:
    received_light_set: tuple[str, ...]
    input_fingerprint: str


def _fail(code: str, message: str) -> None:
    raise CaptureValidationError(code, message)


def validate_capture(
    identity: CaptureIdentity,
    frames: list[CaptureFrame],
    required_lights: frozenset[str] = REQUIRED_LIGHTS,
) -> ValidatedCapture:
    if identity.capture_started_monotonic_ns < identity.start_received_monotonic_ns:
        _fail("CAPTURE_BEFORE_START", "Capture started before the current START was received")
    if (
        identity.previous_camera_trigger_sequence is not None
        and identity.camera_trigger_sequence <= identity.previous_camera_trigger_sequence
    ):
        _fail("STALE_TRIGGER_SEQUENCE", "Camera SDK trigger sequence did not increase")

    light_ids: list[str] = []
    fingerprint_items: list[tuple[str, str, str]] = []
    for frame in frames:
        if frame.capture_id != identity.capture_id:
            _fail("CAPTURE_ID_MISMATCH", "Frame belongs to a different capture")
        if frame.camera_trigger_sequence != identity.camera_trigger_sequence:
            _fail("TRIGGER_SEQUENCE_MISMATCH", "Frame belongs to a different camera trigger")
        if frame.light_id not in required_lights:
            _fail("UNKNOWN_LIGHT", f"Unexpected light channel: {frame.light_id}")
        if frame.light_id in light_ids:
            _fail("DUPLICATE_LIGHT", f"Duplicate light channel: {frame.light_id}")
        light_ids.append(frame.light_id)
        fingerprint_items.append((frame.data_type, frame.light_id, frame.file_hash))

    missing = required_lights.difference(light_ids)
    if missing:
        _fail("MISSING_LIGHT", f"Missing required light channels: {','.join(sorted(missing))}")

    canonical = json.dumps(sorted(fingerprint_items), separators=(",", ":"), ensure_ascii=True)
    fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return ValidatedCapture(tuple(sorted(light_ids)), fingerprint)
