from datetime import datetime, timezone

from sqlalchemy import JSON, BigInteger, Boolean, DateTime, Float, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class InspectionEvent(Base):
    __tablename__ = "inspection_events"
    __table_args__ = (
        UniqueConstraint("handler_id", "handler_session_id", "cycle_id", name="uq_handler_cycle"),
        UniqueConstraint("handler_id", "station", "active_cycle_guard", name="uq_handler_station_active"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    event_uuid: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    source_key_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    trace_id: Mapped[str | None] = mapped_column(String(128), unique=True, index=True)
    handler_id: Mapped[str | None] = mapped_column(String(64), index=True)
    handler_session_id: Mapped[str | None] = mapped_column(String(64))
    cycle_id: Mapped[str | None] = mapped_column(String(64))
    active_cycle_guard: Mapped[str | None] = mapped_column(String(8))
    capture_id: Mapped[str | None] = mapped_column(String(36), unique=True)
    camera_trigger_sequence: Mapped[int | None] = mapped_column(BigInteger)
    start_received_monotonic_ns: Mapped[int | None] = mapped_column(BigInteger)
    capture_started_monotonic_ns: Mapped[int | None] = mapped_column(BigInteger)
    capture_trigger_source: Mapped[str | None] = mapped_column(String(24))
    capture_started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    capture_completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    cycle_deadline_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    device_id: Mapped[str] = mapped_column(String(32), nullable=False)
    device_session_id: Mapped[str] = mapped_column(String(64), nullable=False)
    inspection_sequence: Mapped[str] = mapped_column(String(32), nullable=False)
    product_id: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    batch_id: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    tray_id: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    slot_index: Mapped[str] = mapped_column(String(8), nullable=False)
    station: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    surface: Mapped[str] = mapped_column(String(8), nullable=False, default="TOP")
    association_status: Mapped[str] = mapped_column(String(24), nullable=False)
    required_light_set: Mapped[list[str]] = mapped_column(JSON, default=list)
    received_light_set: Mapped[list[str]] = mapped_column(JSON, default=list)
    ai_decision: Mapped[str | None] = mapped_column(String(16), index=True)
    ai_confidence: Mapped[float | None] = mapped_column(Float)
    defect_code: Mapped[str | None] = mapped_column(String(32))
    reason_code: Mapped[str | None] = mapped_column(String(64))
    image_url: Mapped[str | None] = mapped_column(String(256))
    aoi_bin: Mapped[int | None] = mapped_column(Integer)
    result_category: Mapped[str | None] = mapped_column(String(16))
    human_decision: Mapped[str | None] = mapped_column(String(16))
    final_decision: Mapped[str | None] = mapped_column(String(16))
    decision_source: Mapped[str | None] = mapped_column(String(16))
    handler_publish_status: Mapped[str] = mapped_column(String(32), nullable=False, default="NOT_READY")
    mes_publish_status: Mapped[str] = mapped_column(String(32), nullable=False, default="NOT_READY")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)


class Attachment(Base):
    __tablename__ = "data_attachment_links"
    __table_args__ = (
        UniqueConstraint("event_uuid", "data_type", "light_id", "file_hash"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    event_uuid: Mapped[str | None] = mapped_column(
        ForeignKey("inspection_events.event_uuid", ondelete="CASCADE"), index=True
    )
    quarantine_id: Mapped[str | None] = mapped_column(String(64), index=True)
    light_id: Mapped[str | None] = mapped_column(String(8))
    data_type: Mapped[str] = mapped_column(String(16), nullable=False)
    file_path: Mapped[str] = mapped_column(String(512), nullable=False)
    file_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    checksum_passed: Mapped[bool] = mapped_column(Boolean, default=True)


class InferenceResult(Base):
    __tablename__ = "inference_results"
    __table_args__ = (
        UniqueConstraint("event_uuid", "model_version", "policy_version", "input_fingerprint"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    event_uuid: Mapped[str] = mapped_column(
        ForeignKey("inspection_events.event_uuid", ondelete="CASCADE"), index=True
    )
    model_version: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    policy_version: Mapped[str] = mapped_column(String(32), nullable=False)
    input_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    inference_mode: Mapped[str] = mapped_column(String(16), nullable=False)
    ai_decision: Mapped[str] = mapped_column(String(16), nullable=False)
    ai_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    defect_code: Mapped[str | None] = mapped_column(String(32))
    defect_bbox: Mapped[list[dict[str, object]]] = mapped_column(JSON, default=list)
    measures_3d: Mapped[dict[str, float]] = mapped_column(JSON, default=dict)
    inference_latency_ms: Mapped[int] = mapped_column(Integer, nullable=False)


class QuarantineEvent(Base):
    __tablename__ = "quarantine_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    quarantine_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    source_file_path: Mapped[str] = mapped_column(String(512), nullable=False)
    parse_error: Mapped[str] = mapped_column(String(256), nullable=False)
    extracted_fields: Mapped[dict[str, object]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class HandlerResultOutbox(Base):
    __tablename__ = "handler_result_outbox"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    message_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    event_uuid: Mapped[str] = mapped_column(
        ForeignKey("inspection_events.event_uuid", ondelete="CASCADE"), nullable=False, index=True
    )
    trace_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    cycle_id: Mapped[str] = mapped_column(String(64), nullable=False)
    payload: Mapped[dict[str, object]] = mapped_column(JSON, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="PENDING", index=True)
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    next_retry_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    delivery_uncertain_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_error: Mapped[str | None] = mapped_column(String(512))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)


class MesOutbox(Base):
    __tablename__ = "mes_outbox"
    __table_args__ = (
        UniqueConstraint("event_uuid", "event_revision", name="uq_mes_event_revision"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    message_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    event_uuid: Mapped[str] = mapped_column(
        ForeignKey("inspection_events.event_uuid", ondelete="CASCADE"), nullable=False, index=True
    )
    event_revision: Mapped[int] = mapped_column(Integer, nullable=False)
    event_type: Mapped[str] = mapped_column(String(32), nullable=False)
    payload: Mapped[dict[str, object]] = mapped_column(JSON, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="PENDING", index=True)
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    next_retry_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_error: Mapped[str | None] = mapped_column(String(512))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)
