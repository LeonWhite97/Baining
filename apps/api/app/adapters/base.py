from dataclasses import dataclass
from typing import Mapping, Protocol


@dataclass(frozen=True, slots=True)
class NormalizedAttachment:
    light_id: str
    file_path: str
    file_hash: str


@dataclass(frozen=True, slots=True)
class NormalizedInspection:
    device_id: str
    device_session_id: str
    inspection_sequence: str
    product_id: str
    batch_id: str
    tray_id: str
    slot_index: str
    station: str
    surface: str
    source_key_hash: str
    attachments: tuple[NormalizedAttachment, ...]


class InspectionSourceAdapter(Protocol):
    def normalize(self, raw: Mapping[str, object]) -> NormalizedInspection: ...
