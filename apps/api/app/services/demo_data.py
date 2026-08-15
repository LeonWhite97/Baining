import hashlib
from datetime import datetime, timedelta, timezone
from uuid import NAMESPACE_URL, uuid5

from sqlalchemy import delete
from sqlalchemy.orm import Session

from app.domain.source_key import SourceKeyParts, generate_source_key_hash
from app.models import (
    AnomalyReport,
    Attachment,
    InferenceResult,
    InspectionEvent,
    ModelRelease,
    QuarantineEvent,
    ReviewRecord,
    StationAlert,
)


DEFECTS = ("BALL_BRIDGE", "SOLDER_VOID", "COPLANARITY", "MISSING_BALL")
LIGHTS = ("R", "G", "B", "RING")


def reset_demo(session: Session, seed: int) -> int:
    for model in (
        ReviewRecord,
        AnomalyReport,
        StationAlert,
        ModelRelease,
        InferenceResult,
        Attachment,
        QuarantineEvent,
        InspectionEvent,
    ):
        session.execute(delete(model))

    base_time = datetime(2024, 12, 18, 8, 0, tzinfo=timezone.utc)
    for index in range(1, 25):
        if index <= 16:
            decision, confidence, defect, reason = "PASS", 0.982, None, "POLICY_AUTO_PASS"
        elif index <= 20:
            decision, confidence = "FAIL", 0.91
            defect, reason = DEFECTS[index - 17], "DEFECT_SCORE"
        else:
            decision, confidence, defect, reason = "REVIEW", 0.71, "UNKNOWN", "LOW_CONFIDENCE"
        tray_number = 1 if index <= 12 else 2
        slot_number = index if index <= 12 else index - 12
        station = "ST-02" if 17 <= index <= 22 else "ST-01"
        parts = SourceKeyParts(
            "PIS-01",
            f"BOOT-{seed}",
            str(index),
            f"TRAY-{tray_number:03d}",
            f"{slot_number:02d}",
            "TOP",
        )
        event_uuid = str(uuid5(NAMESPACE_URL, f"pis-in:{seed}:{index}"))
        event = InspectionEvent(
            event_uuid=event_uuid,
            source_key_hash=generate_source_key_hash(parts),
            device_id=parts.device_id,
            device_session_id=parts.device_session_id,
            inspection_sequence=parts.inspection_sequence,
            product_id="BGA-256",
            batch_id=f"LOT-{seed}",
            tray_id=parts.tray_id,
            slot_index=parts.slot_index,
            station=station,
            surface=parts.surface,
            association_status="INFERRED",
            required_light_set=list(LIGHTS),
            received_light_set=list(LIGHTS),
            ai_decision=decision,
            ai_confidence=confidence,
            defect_code=defect,
            reason_code=reason,
            image_url=f"/api/v1/demo/images/{event_uuid}.svg",
            created_at=base_time + timedelta(seconds=index * 8),
        )
        session.add(event)
        # Flush the parent row before adding FK children. This is required by
        # PostgreSQL, which enforces event_uuid immediately during batch inserts.
        session.flush()
        for light in LIGHTS:
            file_hash = hashlib.sha256(f"{event_uuid}:{light}".encode()).hexdigest()
            session.add(
                Attachment(
                    event_uuid=event_uuid,
                    light_id=light,
                    data_type="2D_IMAGE",
                    file_path=f"demo://{event_uuid}/{light}.png",
                    file_hash=file_hash,
                )
            )
        session.add(
            InferenceResult(
                event_uuid=event_uuid,
                model_version="yolov8s-aoi-3.5.2",
                policy_version="policy-3.5.1",
                input_fingerprint=hashlib.sha256(event_uuid.encode()).hexdigest(),
                inference_mode="DEMO",
                ai_decision=decision,
                ai_confidence=confidence,
                defect_code=defect,
                defect_bbox=[] if decision == "PASS" else [{"x": 32, "y": 24, "w": 42, "h": 36}],
                measures_3d={"ball_height_max": 0.43, "coplanarity_max": 0.08},
                inference_latency_ms=24 + index % 9,
            )
        )

    session.add(
        StationAlert(
            alert_id=f"ALT-{seed}-001",
            station="ST-02",
            defect_rate=0.18,
            threshold=0.08,
            sample_count=120,
            status="OPEN",
        )
    )
    session.add_all(
        [
            ModelRelease(
                model_version="yolov8s-aoi-3.5.2",
                status="PRODUCTION",
                metrics={"recall": 0.972, "p95_ms": 46, "false_positive_target": "<=3%"},
            ),
            ModelRelease(
                model_version="yolov8s-aoi-3.6.0-rc1",
                status="SHADOW",
                metrics={"recall": 0.978, "p95_ms": 49, "difference_rate": 0.021},
            ),
        ]
    )
    session.commit()
    return 24
