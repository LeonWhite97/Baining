from typing import Literal

from pydantic import BaseModel, Field


class DemoResetIn(BaseModel):
    seed: int = 202408


class InspectionIn(BaseModel):
    device_id: str
    device_session_id: str
    inspection_sequence: str
    product_id: str
    batch_id: str
    tray_id: str
    slot_index: str
    station: str
    surface: str = "TOP"
    scenario: Literal["NORMAL", "DEFECT", "REVIEW", "MISSING_3D", "MISSING_LIGHT"]


class ReviewDecisionIn(BaseModel):
    event_uuid: str
    decision: Literal["PASS", "FAIL"]
    defect_code: str | None = None
    comment: str = Field(min_length=1, max_length=512)
    reviewer: str = Field(min_length=1, max_length=64)


class OperatorIn(BaseModel):
    operator: str = Field(min_length=1, max_length=64)


class ReportCreateIn(BaseModel):
    alert_id: str
