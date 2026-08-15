import hashlib
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class SourceKeyParts:
    device_id: str
    device_session_id: str
    inspection_sequence: str
    tray_id: str
    slot_index: str
    surface: str = "TOP"


def generate_source_key_hash(parts: SourceKeyParts) -> str:
    canonical = json.dumps(
        asdict(parts),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
