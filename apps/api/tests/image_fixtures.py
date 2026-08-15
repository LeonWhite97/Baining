from hashlib import sha256
from pathlib import Path

from PIL import Image

from app.adapters.base import NormalizedAttachment


LIGHTS = ("R", "G", "B", "RING")


def write_image(
    path: Path,
    *,
    image_format: str = "JPEG",
    size: tuple[int, int] = (64, 48),
) -> str:
    Image.new("RGB", size, (30, 140, 80)).save(path, format=image_format)
    return sha256(path.read_bytes()).hexdigest()


def make_attachments(path: Path, file_hash: str) -> tuple[NormalizedAttachment, ...]:
    return tuple(
        NormalizedAttachment(light_id=light, file_path=str(path), file_hash=file_hash)
        for light in LIGHTS
    )


def make_pis_in_payload(
    path: Path,
    file_hash: str,
    *,
    sequence: str = "1",
    station: str = "ST-PCB",
    scenario: str = "REVIEW",
) -> dict[str, object]:
    return {
        "DeviceID": "PIS-PCB-SIM",
        "DeviceSessionID": "PCB-E2E-BOOT",
        "InspectionSequence": sequence,
        "ProductID": "PCB-STABILITY",
        "BatchID": "PCB-E2E-LOT",
        "TrayID": "PCB-E2E-TRAY",
        "SlotIndex": sequence[-8:],
        "Station": station,
        "Surface": "TOP",
        "Scenario": scenario,
        "Images": [
            {"LightGroup": light, "Path": str(path), "SHA256": file_hash}
            for light in LIGHTS
        ],
    }
