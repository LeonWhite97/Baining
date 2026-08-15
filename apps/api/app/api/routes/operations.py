from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from uuid import NAMESPACE_URL, uuid4, uuid5

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import Response as FastApiResponse
from PIL import Image, UnidentifiedImageError
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db import get_session
from app.adapters.pis_in import IdentityUnavailable, PisInSourceAdapter
from app.domain.source_key import SourceKeyParts, generate_source_key_hash
from app.models import AnomalyReport, InferenceResult, InspectionEvent, ModelRelease, ReviewRecord, StationAlert
from app.models import Attachment, QuarantineEvent
from app.inference.demo import DemoInferenceAdapter
from app.services.inference_orchestrator import run_inference
from app.services.image_evidence import EvidenceValidationError, ValidatedImage, validate_image_set
from app.services.alerting import evaluate_station_alert
from app.schemas.operations import DemoResetIn, InspectionIn, OperatorIn, ReportCreateIn, ReviewDecisionIn
from app.services.demo_data import reset_demo


router = APIRouter()


def quarantine_import(
    session: Session,
    raw: dict[str, object],
    *,
    reason_code: str,
    reason: str,
) -> dict[str, str]:
    quarantine_id = f"QRN-{datetime.now(timezone.utc):%Y%m%d}-{uuid4().hex[:8].upper()}"
    images = raw.get("Images")
    declared_paths = [str(item.get("Path", "")) for item in images if isinstance(item, dict)] if isinstance(images, list) else []
    session.add(
        QuarantineEvent(
            quarantine_id=quarantine_id,
            source_file_path=str(raw.get("SourceFilePath", declared_paths[0] if declared_paths else "pis-in://unidentified")),
            parse_error=f"{reason_code}: {reason}",
            extracted_fields={
                **{key: value for key, value in raw.items() if key in {"DeviceID", "TrayID", "SlotIndex", "InspectionSequence"}},
                "declared_paths": declared_paths,
            },
        )
    )
    session.commit()
    return {
        "status": "QUARANTINED",
        "quarantine_id": quarantine_id,
        "reason_code": reason_code,
        "reason": reason,
    }


def _attachment_identity(images: tuple[ValidatedImage, ...]) -> tuple[tuple[str, str, str], ...]:
    return tuple(sorted((item.light_id, str(item.path), item.sha256) for item in images))


def _stored_attachment_identity(session: Session, event_uuid: str) -> tuple[tuple[str, str, str], ...]:
    rows = session.scalars(select(Attachment).where(Attachment.event_uuid == event_uuid)).all()
    return tuple(sorted((str(item.light_id), str(Path(item.file_path).resolve()), item.file_hash.lower()) for item in rows))


def event_payload(event: InspectionEvent, result: InferenceResult | None = None) -> dict[str, object]:
    return {
        "event_uuid": event.event_uuid,
        "device_id": event.device_id,
        "product_id": event.product_id,
        "batch_id": event.batch_id,
        "tray_id": event.tray_id,
        "slot_index": event.slot_index,
        "station": event.station,
        "decision": event.ai_decision,
        "confidence": event.ai_confidence,
        "defect_code": event.defect_code,
        "reason_code": event.reason_code,
        "image_url": event.image_url,
        "bbox": result.defect_bbox if result else [],
        "measures_3d": result.measures_3d if result else {},
        "model_version": result.model_version if result else None,
        "created_at": event.created_at.isoformat(),
    }


@router.post("/demo/reset")
def demo_reset(payload: DemoResetIn, request: Request, session: Session = Depends(get_session)) -> dict[str, int]:
    if request.app.state.mode != "demo":
        raise HTTPException(status_code=404)
    return {"seed": payload.seed, "events": reset_demo(session, payload.seed)}


@router.get("/dashboard/summary")
def dashboard_summary(session: Session = Depends(get_session)) -> dict[str, object]:
    rows = session.scalars(select(InspectionEvent).order_by(InspectionEvent.created_at)).all()
    decisions = {name: sum(row.ai_decision == name.upper() for row in rows) for name in ("pass", "fail", "review")}
    trend = [
        {"time": row.created_at.strftime("%H:%M:%S"), "decision": row.ai_decision}
        for row in rows[-12:]
    ]
    open_alerts = session.scalar(select(func.count()).select_from(StationAlert).where(StationAlert.status != "CLOSED")) or 0
    return {"counts": {"total": len(rows), **decisions}, "open_alerts": open_alerts, "defect_trend": trend}


@router.get("/inspections")
def inspections(decision: str | None = None, session: Session = Depends(get_session)) -> dict[str, object]:
    statement = select(InspectionEvent).order_by(InspectionEvent.created_at.desc())
    if decision:
        statement = statement.where(InspectionEvent.ai_decision == decision.upper())
    rows = session.scalars(statement).all()
    return {"items": [event_payload(row) for row in rows], "total": len(rows)}


@router.get("/inspections/{event_uuid}")
def inspection_detail(event_uuid: str, session: Session = Depends(get_session)) -> dict[str, object]:
    event = session.scalar(select(InspectionEvent).where(InspectionEvent.event_uuid == event_uuid))
    if not event:
        raise HTTPException(status_code=404, detail="Inspection event not found")
    result = session.scalar(select(InferenceResult).where(InferenceResult.event_uuid == event_uuid))
    payload = event_payload(event, result)
    payload["attachment_count"] = session.scalar(
        select(func.count()).select_from(Attachment).where(Attachment.event_uuid == event_uuid)
    ) or 0
    payload["inference_result_count"] = session.scalar(
        select(func.count()).select_from(InferenceResult).where(InferenceResult.event_uuid == event_uuid)
    ) or 0
    return payload


@router.get("/inspections/{event_uuid}/image", include_in_schema=False)
def inspection_image(
    event_uuid: str,
    request: Request,
    light_id: str = "RING",
    session: Session = Depends(get_session),
) -> FastApiResponse:
    event = session.scalar(select(InspectionEvent).where(InspectionEvent.event_uuid == event_uuid))
    if event is None:
        raise HTTPException(status_code=404, detail="Inspection event not found")
    attachment = session.scalar(
        select(Attachment).where(
            Attachment.event_uuid == event_uuid,
            Attachment.data_type == "2D_IMAGE",
            Attachment.light_id == light_id,
        )
    )
    if attachment is None:
        raise HTTPException(status_code=404, detail="Image evidence not found")

    path = Path(attachment.file_path).resolve(strict=False)
    if not path.is_relative_to(request.app.state.image_root) or not path.is_file():
        raise HTTPException(status_code=409, detail="Image evidence is unavailable")
    content = path.read_bytes()
    if sha256(content).hexdigest() != attachment.file_hash.lower():
        raise HTTPException(status_code=409, detail="Image evidence checksum changed")

    try:
        with Image.open(BytesIO(content)) as image:
            media_type = {"JPEG": "image/jpeg", "PNG": "image/png"}.get(image.format)
    except (OSError, UnidentifiedImageError):
        media_type = None
    if media_type is None:
        raise HTTPException(status_code=409, detail="Image evidence format changed")
    return FastApiResponse(content=content, media_type=media_type)


@router.post("/inspections", status_code=status.HTTP_201_CREATED)
def create_inspection(payload: InspectionIn, response: Response, session: Session = Depends(get_session)) -> dict[str, object]:
    parts = SourceKeyParts(
        payload.device_id,
        payload.device_session_id,
        payload.inspection_sequence,
        payload.tray_id,
        payload.slot_index,
        payload.surface,
    )
    source_hash = generate_source_key_hash(parts)
    existing = session.scalar(select(InspectionEvent).where(InspectionEvent.source_key_hash == source_hash))
    if existing:
        response.status_code = status.HTTP_200_OK
        return event_payload(existing)
    mapping = {
        "NORMAL": ("PASS", 0.985, None, "POLICY_AUTO_PASS"),
        "DEFECT": ("FAIL", 0.91, "BALL_BRIDGE", "DEFECT_SCORE"),
        "REVIEW": ("REVIEW", 0.71, "UNKNOWN", "LOW_CONFIDENCE"),
        "MISSING_3D": ("REVIEW", 0.95, None, "INPUT_INCOMPLETE"),
        "MISSING_LIGHT": ("REVIEW", 0.95, None, "INPUT_INCOMPLETE"),
    }
    decision, confidence, defect_code, reason_code = mapping[payload.scenario]
    event_uuid = str(uuid5(NAMESPACE_URL, f"pis-in:{source_hash}"))
    event = InspectionEvent(
        event_uuid=event_uuid,
        source_key_hash=source_hash,
        device_id=payload.device_id,
        device_session_id=payload.device_session_id,
        inspection_sequence=payload.inspection_sequence,
        product_id=payload.product_id,
        batch_id=payload.batch_id,
        tray_id=payload.tray_id,
        slot_index=payload.slot_index,
        station=payload.station,
        surface=payload.surface,
        association_status="INFERRED",
        required_light_set=["R", "G", "B", "RING"],
        received_light_set=["R", "G", "B", "RING"] if payload.scenario != "MISSING_LIGHT" else ["R", "G", "B"],
        ai_decision=decision,
        ai_confidence=confidence,
        defect_code=defect_code,
        reason_code=reason_code,
        image_url=f"/api/v1/demo/images/{event_uuid}.svg",
    )
    session.add(event)
    session.commit()
    evaluate_station_alert(session, payload.station)
    return event_payload(event)


@router.post("/inspections/import/pis-in")
def import_pis_in(raw: dict[str, object], response: Response, session: Session = Depends(get_session)) -> dict[str, object]:
    adapter = PisInSourceAdapter()
    try:
        normalized = adapter.normalize(raw)
    except IdentityUnavailable as exc:
        response.status_code = status.HTTP_202_ACCEPTED
        return quarantine_import(
            session, raw, reason_code="IDENTITY_MISSING", reason=str(exc)
        )
    except ValueError as exc:
        response.status_code = status.HTTP_202_ACCEPTED
        return quarantine_import(
            session, raw, reason_code="LIGHT_SET_INVALID", reason=str(exc)
        )

    try:
        validated_images = validate_image_set(normalized.attachments, Path(session.info["image_root"]))
    except KeyError:
        raise RuntimeError("Image root is not configured for this request") from None
    except EvidenceValidationError as exc:
        response.status_code = status.HTTP_202_ACCEPTED
        return quarantine_import(
            session, raw, reason_code=exc.reason_code, reason=exc.public_reason
        )

    existing = session.scalar(select(InspectionEvent).where(InspectionEvent.source_key_hash == normalized.source_key_hash))
    if existing:
        if _stored_attachment_identity(session, existing.event_uuid) != _attachment_identity(validated_images):
            response.status_code = status.HTTP_202_ACCEPTED
            return quarantine_import(
                session,
                raw,
                reason_code="IDEMPOTENCY_CONFLICT",
                reason="Source identity already exists with different image evidence",
            )
        response.status_code = status.HTTP_200_OK
        attachments_count = session.scalar(select(func.count()).select_from(Attachment).where(Attachment.event_uuid == existing.event_uuid)) or 0
        payload = event_payload(existing)
        payload["attachment_count"] = attachments_count
        return payload
    event_uuid = str(uuid5(NAMESPACE_URL, f"pis-in:{normalized.source_key_hash}"))
    scenario = str(raw.get("Scenario", "REVIEW")) if session.info["mode"] == "demo" else "REVIEW"
    input_complete = scenario not in {"MISSING_3D", "MISSING_LIGHT"} and len(normalized.attachments) >= 4
    output = run_inference(DemoInferenceAdapter(), event_uuid=event_uuid, scenario=scenario, input_complete=input_complete)
    event = InspectionEvent(
        event_uuid=event_uuid, source_key_hash=normalized.source_key_hash, device_id=normalized.device_id,
        device_session_id=normalized.device_session_id, inspection_sequence=normalized.inspection_sequence,
        product_id=normalized.product_id, batch_id=normalized.batch_id, tray_id=normalized.tray_id,
        slot_index=normalized.slot_index, station=normalized.station, surface=normalized.surface,
        association_status="INFERRED", required_light_set=["R", "G", "B", "RING"],
        received_light_set=[item.light_id for item in normalized.attachments], ai_decision=output.decision,
        ai_confidence=output.confidence, defect_code=output.defect_code, reason_code=output.reason_code,
        image_url=f"/api/v1/inspections/{event_uuid}/image?light_id=RING",
    )
    try:
        session.add(event)
        session.flush()
        for item in validated_images:
            session.add(Attachment(event_uuid=event_uuid, light_id=item.light_id, data_type="2D_IMAGE", file_path=str(item.path), file_hash=item.sha256))
        inference_output = output.output
        session.add(InferenceResult(
            event_uuid=event_uuid, model_version=inference_output.model_version, policy_version="policy-3.5.1",
            input_fingerprint=normalized.source_key_hash, inference_mode=output.mode.value, ai_decision=output.decision,
            ai_confidence=output.confidence, defect_code=output.defect_code,
            defect_bbox=[{"x": box[0], "y": box[1], "w": box[2], "h": box[3]} for box in inference_output.boxes],
            measures_3d={}, inference_latency_ms=inference_output.latency_ms,
        ))
        session.commit()
    except IntegrityError:
        session.rollback()
        existing = session.scalar(select(InspectionEvent).where(InspectionEvent.source_key_hash == normalized.source_key_hash))
        if existing and _stored_attachment_identity(session, existing.event_uuid) == _attachment_identity(validated_images):
            response.status_code = status.HTTP_200_OK
            payload = event_payload(existing)
            payload["attachment_count"] = session.scalar(
                select(func.count()).select_from(Attachment).where(Attachment.event_uuid == existing.event_uuid)
            ) or 0
            return payload
        if existing:
            response.status_code = status.HTTP_202_ACCEPTED
            return quarantine_import(
                session,
                raw,
                reason_code="IDEMPOTENCY_CONFLICT",
                reason="Source identity already exists with different image evidence",
            )
        raise
    evaluate_station_alert(session, normalized.station)
    response.status_code = status.HTTP_201_CREATED
    payload = event_payload(event)
    payload["attachment_count"] = len(normalized.attachments)
    return payload


@router.get("/trays/{tray_id}")
def tray(tray_id: str, session: Session = Depends(get_session)) -> dict[str, object]:
    rows = session.scalars(
        select(InspectionEvent).where(InspectionEvent.tray_id == tray_id).order_by(InspectionEvent.slot_index)
    ).all()
    if not rows:
        raise HTTPException(status_code=404, detail="Tray not found")
    return {"tray_id": tray_id, "slots": [event_payload(row) for row in rows]}


@router.get("/reviews")
def review_queue(session: Session = Depends(get_session)) -> dict[str, object]:
    reviewed = select(ReviewRecord.event_uuid)
    rows = session.scalars(
        select(InspectionEvent)
        .where(InspectionEvent.ai_decision == "REVIEW", InspectionEvent.event_uuid.not_in(reviewed))
        .order_by(InspectionEvent.created_at)
    ).all()
    return {"items": [event_payload(row) for row in rows], "total": len(rows)}


@router.post("/reviews", status_code=status.HTTP_201_CREATED)
def create_review(payload: ReviewDecisionIn, session: Session = Depends(get_session)) -> dict[str, object]:
    event = session.scalar(select(InspectionEvent).where(InspectionEvent.event_uuid == payload.event_uuid))
    if not event:
        raise HTTPException(status_code=404, detail="Inspection event not found")
    if session.scalar(select(ReviewRecord).where(ReviewRecord.event_uuid == payload.event_uuid)):
        raise HTTPException(status_code=409, detail="Event already reviewed")
    record = ReviewRecord(**payload.model_dump(), golden_status="CONFIRMED")
    event.ai_decision = payload.decision
    event.defect_code = payload.defect_code
    event.reason_code = "MANUAL_REVIEW"
    session.add(record)
    session.commit()
    return {"review_id": record.id, "event_uuid": record.event_uuid, "golden_status": record.golden_status}


def alert_payload(alert: StationAlert) -> dict[str, object]:
    return {
        "alert_id": alert.alert_id,
        "station": alert.station,
        "defect_rate": alert.defect_rate,
        "threshold": alert.threshold,
        "sample_count": alert.sample_count,
        "status": alert.status,
        "acknowledged_by": alert.acknowledged_by,
    }


@router.get("/alerts")
def alerts(session: Session = Depends(get_session)) -> dict[str, object]:
    rows = session.scalars(select(StationAlert).order_by(StationAlert.created_at.desc())).all()
    return {"items": [alert_payload(row) for row in rows], "total": len(rows)}


@router.post("/alerts/{alert_id}/acknowledge")
def acknowledge_alert(alert_id: str, payload: OperatorIn, session: Session = Depends(get_session)) -> dict[str, object]:
    alert = session.scalar(select(StationAlert).where(StationAlert.alert_id == alert_id))
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    alert.status = "ACKNOWLEDGED"
    alert.acknowledged_by = payload.operator
    alert.acknowledged_at = datetime.now(timezone.utc)
    session.commit()
    return alert_payload(alert)


@router.post("/alerts/{alert_id}/close")
def close_alert(alert_id: str, payload: OperatorIn, session: Session = Depends(get_session)) -> dict[str, object]:
    alert = session.scalar(select(StationAlert).where(StationAlert.alert_id == alert_id))
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    alert.status = "CLOSED"
    alert.closed_by = payload.operator
    alert.closed_at = datetime.now(timezone.utc)
    session.commit()
    return alert_payload(alert)


def report_payload(report: AnomalyReport) -> dict[str, object]:
    return {
        "report_id": report.report_id,
        "alert_id": report.alert_id,
        "status": report.status,
        "summary": report.summary,
        "observed_facts": report.observed_facts,
        "open_questions": report.open_questions,
        "event_uuids": report.event_uuids,
        "agent_status": report.agent_status,
    }


@router.post("/reports", status_code=status.HTTP_201_CREATED)
def create_report(payload: ReportCreateIn, request: Request, session: Session = Depends(get_session)) -> dict[str, object]:
    alert = session.scalar(select(StationAlert).where(StationAlert.alert_id == payload.alert_id))
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    if alert.status == "OPEN":
        raise HTTPException(status_code=409, detail="Alert must be acknowledged first")
    events = session.scalars(
        select(InspectionEvent).where(InspectionEvent.station == alert.station, InspectionEvent.ai_decision == "FAIL")
    ).all()
    observed_facts = [
        f"工站 {alert.station} 当前缺陷率 {alert.defect_rate:.1%}，阈值 {alert.threshold:.1%}",
        f"统计窗口样本量 {alert.sample_count}",
    ]
    agent_status = "NOT_CONFIGURED"
    agent_client = getattr(request.app.state, "agent_client", None)
    if agent_client is not None:
        try:
            agent_draft = agent_client.draft_report(
                {
                    "station": alert.station,
                    "defect_rate": alert.defect_rate,
                    "threshold": alert.threshold,
                    "sample_count": alert.sample_count,
                    "event_uuids": [event.event_uuid for event in events],
                }
            )
            observed_facts = list(agent_draft.get("observed_facts") or observed_facts)
            agent_status = "SUCCEEDED"
        except (TimeoutError, OSError, ValueError):
            agent_status = "UNAVAILABLE"
    report = AnomalyReport(
        report_id=f"RPT-{uuid4()}",
        alert_id=alert.alert_id,
        status="DRAFT",
        summary=f"{alert.station} 缺陷率超过阈值，需确认设备、物料与工艺变化。",
        observed_facts=observed_facts,
        open_questions=["是否发生换线或参数调整", "是否存在同批次物料集中异常"],
        event_uuids=[event.event_uuid for event in events],
        agent_status=agent_status,
    )
    session.add(report)
    session.commit()
    return report_payload(report)


@router.get("/reports")
def reports(session: Session = Depends(get_session)) -> dict[str, object]:
    rows = session.scalars(select(AnomalyReport).order_by(AnomalyReport.created_at.desc())).all()
    return {"items": [report_payload(row) for row in rows], "total": len(rows)}


@router.get("/reports/{report_id}")
def report_detail(report_id: str, session: Session = Depends(get_session)) -> dict[str, object]:
    report = session.scalar(select(AnomalyReport).where(AnomalyReport.report_id == report_id))
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    return report_payload(report)


@router.get("/model-releases")
def model_releases(session: Session = Depends(get_session)) -> dict[str, object]:
    rows = session.scalars(select(ModelRelease).order_by(ModelRelease.created_at.desc())).all()
    return {"items": [{"model_version": row.model_version, "status": row.status, "metrics": row.metrics} for row in rows]}


@router.get("/project-profile")
def project_profile() -> dict[str, object]:
    return {
        "name": "PIS-IN AOI AI 智能质检系统",
        "version": "V3.5 落地与展示增强版",
        "period": "2024.09-2025.01",
        "team_count": 8,
        "team": ["产品/AI负责人 1", "后端 2", "前端 1", "算法 2", "测试 1", "实施运维 1"],
        "agents": ["数据质量 Agent", "复核与异常报告 Agent", "模型治理 Agent"],
        "quality_targets": {
            "baseline": "12%",
            "poc": "<=6%",
            "controlled_rollout": "<=3%",
            "mature": "<=1.5%",
            "full_inspection": "<=0.5%",
        },
        "compute": ["训练：2 x NVIDIA L40S 48GB", "边缘：2 x RTX 4000 Ada 20GB", "Agent/RAG：1 x NVIDIA L4 24GB"],
    }


@router.get("/demo/images/{event_uuid}.svg", include_in_schema=False)
def demo_image(event_uuid: str) -> FastApiResponse:
    hue = int(event_uuid.replace("-", "")[:2], 16)
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="960" height="640" viewBox="0 0 960 640">
<rect width="960" height="640" fill="#172126"/><g stroke="#35515b" stroke-width="2" fill="#24343a">
<rect x="96" y="80" width="768" height="480" rx="12"/><circle cx="300" cy="240" r="52"/><circle cx="480" cy="240" r="52"/><circle cx="660" cy="240" r="52"/><circle cx="300" cy="410" r="52"/><circle cx="480" cy="410" r="52"/><circle cx="660" cy="410" r="52"/></g>
<rect x="{240 + hue}" y="180" width="110" height="92" fill="none" stroke="#ef5b5b" stroke-width="5"/><text x="36" y="606" fill="#9eb2ba" font-family="monospace" font-size="22">AOI DEMO {event_uuid[:8]}</text></svg>'''
    return FastApiResponse(content=svg, media_type="image/svg+xml")
