from dataclasses import dataclass
from time import monotonic_ns
from uuid import NAMESPACE_URL, uuid5

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.domain.handler_cycle import (
    ACTIVE_ASSOCIATION_STATUSES,
    HandlerStartEvidence,
    event_matches_evidence,
)
from app.models import InspectionEvent
from app.schemas.handler import AOIStartRequest


class HandlerCycleConflict(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True, slots=True)
class StartCycleResult:
    event: InspectionEvent
    created: bool


def start_cycle(session: Session, request: AOIStartRequest) -> StartCycleResult:
    evidence = HandlerStartEvidence.from_request(request)
    existing = session.scalar(select(InspectionEvent).where(InspectionEvent.trace_id == request.trace_id))
    if existing is not None:
        if event_matches_evidence(existing, evidence):
            return StartCycleResult(existing, False)
        raise HandlerCycleConflict("EVIDENCE_CONFLICT", "TraceID already exists with different evidence")

    active = session.scalar(
        select(InspectionEvent)
        .where(
            InspectionEvent.handler_id == request.handler_id,
            InspectionEvent.station == request.station_code,
            InspectionEvent.association_status.in_(ACTIVE_ASSOCIATION_STATUSES),
        )
        .order_by(InspectionEvent.created_at.desc())
        .limit(1)
    )
    if active is not None:
        raise HandlerCycleConflict("ACTIVE_CYCLE_EXISTS", "Handler has an unfinished AOI cycle")

    source_hash = evidence.identity_hash()
    event_uuid = str(uuid5(NAMESPACE_URL, f"handler-aoi:{source_hash}"))
    event = InspectionEvent(
        event_uuid=event_uuid,
        source_key_hash=source_hash,
        trace_id=request.trace_id,
        handler_id=request.handler_id,
        handler_session_id=request.handler_session_id,
        cycle_id=request.cycle_id,
        active_cycle_guard="ACTIVE",
        start_received_monotonic_ns=monotonic_ns(),
        device_id=request.handler_id,
        device_session_id=request.handler_session_id,
        inspection_sequence=request.cycle_id,
        product_id=request.product_id,
        batch_id=request.batch_id,
        tray_id=request.tray_id,
        slot_index=request.slot_index,
        station=request.station_code,
        surface=request.surface,
        association_status="START_RECEIVED",
        required_light_set=["R", "G", "B", "RING"],
        received_light_set=[],
        handler_publish_status="NOT_READY",
        mes_publish_status="NOT_READY",
    )
    session.add(event)
    try:
        session.commit()
    except IntegrityError as exc:
        session.rollback()
        raced = session.scalar(select(InspectionEvent).where(InspectionEvent.trace_id == request.trace_id))
        if raced is not None and event_matches_evidence(raced, evidence):
            return StartCycleResult(raced, False)
        raced_active = session.scalar(
            select(InspectionEvent).where(
                InspectionEvent.handler_id == request.handler_id,
                InspectionEvent.station == request.station_code,
                InspectionEvent.active_cycle_guard == "ACTIVE",
            )
        )
        if raced_active is not None:
            raise HandlerCycleConflict("ACTIVE_CYCLE_EXISTS", "Handler has an unfinished AOI cycle") from exc
        raise HandlerCycleConflict("CONCURRENT_CYCLE_CONFLICT", "Cycle creation conflicted with another request") from exc
    return StartCycleResult(event, True)
