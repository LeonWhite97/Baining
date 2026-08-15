from typing import Mapping

from app.adapters.base import NormalizedAttachment, NormalizedInspection
from app.domain.source_key import SourceKeyParts, generate_source_key_hash


class IdentityUnavailable(ValueError):
    pass


class PisInSourceAdapter:
    required_fields = (
        "DeviceID",
        "DeviceSessionID",
        "InspectionSequence",
        "ProductID",
        "BatchID",
        "TrayID",
        "SlotIndex",
        "Station",
    )

    def normalize(self, raw: Mapping[str, object]) -> NormalizedInspection:
        for field in self.required_fields:
            if not raw.get(field):
                raise IdentityUnavailable(f"Missing identity field: {field}")
        parts = SourceKeyParts(
            device_id=str(raw["DeviceID"]),
            device_session_id=str(raw["DeviceSessionID"]),
            inspection_sequence=str(raw["InspectionSequence"]),
            tray_id=str(raw["TrayID"]),
            slot_index=str(raw["SlotIndex"]),
            surface=str(raw.get("Surface", "TOP")),
        )
        images = raw.get("Images", [])
        if not isinstance(images, list):
            raise ValueError("Images must be a list")
        try:
            attachments = tuple(
                NormalizedAttachment(
                    light_id=str(item["LightGroup"]),
                    file_path=str(item["Path"]),
                    file_hash=str(item["SHA256"]),
                )
                for item in images
                if isinstance(item, dict)
            )
        except KeyError as exc:
            raise ValueError(f"Image attachment is missing field: {exc.args[0]}") from None
        return NormalizedInspection(
            device_id=parts.device_id,
            device_session_id=parts.device_session_id,
            inspection_sequence=parts.inspection_sequence,
            product_id=str(raw["ProductID"]),
            batch_id=str(raw["BatchID"]),
            tray_id=parts.tray_id,
            slot_index=parts.slot_index,
            station=str(raw["Station"]),
            surface=parts.surface,
            source_key_hash=generate_source_key_hash(parts),
            attachments=attachments,
        )
