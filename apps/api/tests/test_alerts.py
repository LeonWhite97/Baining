from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from fastapi.testclient import TestClient

from app.main import create_app
from app.models import InspectionEvent, StationAlert
from app.services.alerting import evaluate_station_alert


def test_alerting_requires_minimum_sample_and_uses_fail_rate() -> None:
    app = create_app(database_url="sqlite+pysqlite:///:memory:", mode="demo")
    with TestClient(app), app.state.session_factory() as session:
        now = datetime.now(timezone.utc)
        for index in range(10):
            session.add(
                InspectionEvent(
                    event_uuid=f"alert-event-{index}", source_key_hash=f"alert-hash-{index}", device_id="PIS-01",
                    device_session_id="BOOT-A", inspection_sequence=str(index), product_id="BGA-256", batch_id="LOT-A",
                    tray_id="TRAY-A", slot_index=str(index), station="ST-99", surface="TOP", association_status="INFERRED",
                    required_light_set=[], received_light_set=[], ai_decision="FAIL" if index < 2 else "PASS",
                    ai_confidence=0.9, reason_code="TEST", image_url="demo://test", created_at=now - timedelta(seconds=index),
                )
            )
        session.commit()
        alert = evaluate_station_alert(session, "ST-99", window_size=10, threshold=0.1, min_samples=10)
        assert alert is not None
        assert alert.status == "OPEN"
        assert alert.defect_rate == 0.2
        assert session.scalar(select(StationAlert).where(StationAlert.alert_id == alert.alert_id)) is not None
