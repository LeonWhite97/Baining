from datetime import datetime

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base
from app.models.inspection import utc_now


class ReviewRecord(Base):
    __tablename__ = "review_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    event_uuid: Mapped[str] = mapped_column(
        ForeignKey("inspection_events.event_uuid", ondelete="CASCADE"), unique=True, index=True
    )
    decision: Mapped[str] = mapped_column(String(16), nullable=False)
    defect_code: Mapped[str | None] = mapped_column(String(32))
    comment: Mapped[str] = mapped_column(String(512), nullable=False)
    reviewer: Mapped[str] = mapped_column(String(64), nullable=False)
    golden_status: Mapped[str] = mapped_column(String(16), default="CONFIRMED")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class StationAlert(Base):
    __tablename__ = "station_alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    alert_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    station: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    defect_rate: Mapped[float] = mapped_column(Float, nullable=False)
    threshold: Mapped[float] = mapped_column(Float, nullable=False)
    sample_count: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    acknowledged_by: Mapped[str | None] = mapped_column(String(64))
    acknowledged_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    closed_by: Mapped[str | None] = mapped_column(String(64))
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class AnomalyReport(Base):
    __tablename__ = "anomaly_reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    report_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    alert_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    summary: Mapped[str] = mapped_column(String(512), nullable=False)
    observed_facts: Mapped[list[str]] = mapped_column(JSON, default=list)
    open_questions: Mapped[list[str]] = mapped_column(JSON, default=list)
    event_uuids: Mapped[list[str]] = mapped_column(JSON, default=list)
    agent_status: Mapped[str] = mapped_column(String(24), nullable=False, default="NOT_CONFIGURED")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class ModelRelease(Base):
    __tablename__ = "model_releases"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_version: Mapped[str] = mapped_column(String(32), unique=True, nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False)
    metrics: Mapped[dict[str, object]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
