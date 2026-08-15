from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import InspectionEvent, StationAlert


def evaluate_station_alert(
    session: Session,
    station: str,
    *,
    window_size: int = 100,
    threshold: float = 0.08,
    min_samples: int = 20,
) -> StationAlert | None:
    rows = session.scalars(
        select(InspectionEvent).where(InspectionEvent.station == station).order_by(InspectionEvent.created_at.desc()).limit(window_size)
    ).all()
    if len(rows) < min_samples:
        return None
    defect_rate = sum(row.ai_decision == "FAIL" for row in rows) / len(rows)
    if defect_rate < threshold:
        return None
    existing = session.scalar(select(StationAlert).where(StationAlert.station == station, StationAlert.status == "OPEN"))
    if existing:
        return existing
    alert = StationAlert(
        alert_id=f"ALT-{station}-{uuid4().hex[:8]}", station=station, defect_rate=defect_rate,
        threshold=threshold, sample_count=len(rows), status="OPEN", created_at=datetime.now(timezone.utc),
    )
    session.add(alert)
    session.commit()
    return alert
