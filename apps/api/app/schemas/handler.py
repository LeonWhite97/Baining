from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class AOIStartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trace_id: str = Field(pattern=r"^[A-Za-z0-9_.-]{1,128}$")
    handler_id: str = Field(min_length=1, max_length=64)
    handler_session_id: str = Field(min_length=1, max_length=64)
    cycle_id: str = Field(min_length=1, max_length=64)
    station_code: Literal["AOI"] = "AOI"
    product_id: str = Field(min_length=1, max_length=32)
    batch_id: str = Field(min_length=1, max_length=32)
    tray_id: str = Field(min_length=1, max_length=32)
    slot_index: str = Field(pattern=r"^[A-Za-z0-9_.-]{1,8}$")
    surface: Literal["TOP", "BOTTOM"] = "TOP"
