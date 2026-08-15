from sqlalchemy import create_engine, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
import pytest

from app.models import Base, HandlerResultOutbox, InspectionEvent, MesOutbox


def build_start_event(*, trace_id: str = "TRACE-1") -> InspectionEvent:
    return InspectionEvent(
        event_uuid=f"evt-{trace_id}",
        source_key_hash=(trace_id.encode().hex() + "0" * 64)[:64],
        trace_id=trace_id,
        handler_id="HANDLER-1",
        handler_session_id="SESSION-1",
        cycle_id=f"CYCLE-{trace_id}",
        device_id="HANDLER-1",
        device_session_id="SESSION-1",
        inspection_sequence=f"CYCLE-{trace_id}",
        product_id="BGA-256",
        batch_id="LOT-1",
        tray_id="TRAY-1",
        slot_index="01",
        station="AOI",
        surface="TOP",
        association_status="START_RECEIVED",
    )


def test_start_stage_event_allows_missing_inference_fields() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        event = build_start_event()
        session.add(event)
        session.commit()

        stored = session.scalar(select(InspectionEvent).where(InspectionEvent.trace_id == "TRACE-1"))

    assert stored is not None
    assert stored.ai_decision is None
    assert stored.ai_confidence is None
    assert stored.reason_code is None
    assert stored.image_url is None
    assert stored.handler_publish_status == "NOT_READY"
    assert stored.mes_publish_status == "NOT_READY"


def test_result_outboxes_have_independent_status() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        event = build_start_event(trace_id="TRACE-2")
        session.add(event)
        session.flush()
        session.add(
            HandlerResultOutbox(
                message_id="handler-message-1",
                event_uuid=event.event_uuid,
                trace_id=event.trace_id,
                cycle_id=event.cycle_id,
                payload={"aoi_bin": 201},
            )
        )
        session.add(
            MesOutbox(
                message_id="mes-message-1",
                event_uuid=event.event_uuid,
                event_revision=1,
                event_type="AOI_RESULT",
                payload={"aoi_bin": 201},
            )
        )
        session.commit()

        handler_outbox = session.scalar(select(HandlerResultOutbox))
        mes_outbox = session.scalar(select(MesOutbox))

    assert handler_outbox is not None
    assert handler_outbox.status == "PENDING"
    assert mes_outbox is not None
    assert mes_outbox.status == "PENDING"


def test_capture_identity_fields_are_persisted_for_audit() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        event = build_start_event(trace_id="TRACE-3")
        event.capture_id = "f67af360-41a6-4a6c-9d15-7a9e6b28f194"
        event.camera_trigger_sequence = 101
        event.start_received_monotonic_ns = 1_000
        event.capture_started_monotonic_ns = 1_100
        event.capture_trigger_source = "HANDLER_START"
        session.add(event)
        session.commit()

    with Session(engine) as session:
        stored = session.scalar(select(InspectionEvent).where(InspectionEvent.trace_id == "TRACE-3"))

    assert stored is not None
    assert stored.start_received_monotonic_ns == 1_000
    assert stored.capture_started_monotonic_ns == 1_100
    assert stored.capture_trigger_source == "HANDLER_START"


def test_database_rejects_two_active_cycles_for_one_handler_station() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        first = build_start_event(trace_id="TRACE-ACTIVE-1")
        first.active_cycle_guard = "ACTIVE"
        session.add(first)
        session.commit()

    with Session(engine) as session:
        second = build_start_event(trace_id="TRACE-ACTIVE-2")
        second.active_cycle_guard = "ACTIVE"
        session.add(second)
        with pytest.raises(IntegrityError):
            session.commit()
