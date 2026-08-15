from enum import StrEnum


class ScenarioKind(StrEnum):
    NORMAL = "NORMAL"
    DEFECT = "DEFECT"
    REVIEW = "REVIEW"
    MISSING_3D = "MISSING_3D"
    MISSING_LIGHT = "MISSING_LIGHT"
    STATION_SPIKE = "STATION_SPIKE"


def build_scenario(seed: int, scenario: ScenarioKind) -> dict[str, str]:
    tray_number = seed // 12 + 1
    slot_number = seed % 12 + 1
    return {
        "device_id": "PIS-SIM-01",
        "device_session_id": "SIM-BOOT-202408",
        "inspection_sequence": str(seed),
        "product_id": "BGA-256",
        "batch_id": "LOT-SIM-202408",
        "tray_id": f"SIM-TRAY-{tray_number:03d}",
        "slot_index": f"{slot_number:02d}",
        "station": "ST-02" if scenario is ScenarioKind.STATION_SPIKE else "ST-01",
        "surface": "TOP",
        "scenario": "DEFECT" if scenario is ScenarioKind.STATION_SPIKE else scenario.value,
    }
