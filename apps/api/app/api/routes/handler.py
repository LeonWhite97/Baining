from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy.orm import Session

from app.db import get_session
from app.models import InspectionEvent
from app.schemas.handler import AOIStartRequest
from app.services.handler_cycles import HandlerCycleConflict, start_cycle


router = APIRouter()


def _start_payload(event: InspectionEvent) -> dict[str, object]:
    return {
        "event_uuid": event.event_uuid,
        "trace_id": event.trace_id,
        "cycle_id": event.cycle_id,
        "association_status": event.association_status,
        "ai_decision": event.ai_decision,
        "handler_publish_status": event.handler_publish_status,
    }


@router.post("/inspections/aoi/start", status_code=status.HTTP_201_CREATED)
def handler_start(
    payload: AOIStartRequest,
    response: Response,
    session: Session = Depends(get_session),
) -> dict[str, object]:
    try:
        result = start_cycle(session, payload)
    except HandlerCycleConflict as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": exc.code, "message": exc.message},
        ) from exc
    if not result.created:
        response.status_code = status.HTTP_200_OK
    return _start_payload(result.event)
