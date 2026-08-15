import hashlib
import json
from dataclasses import asdict, dataclass

from app.schemas.handler import AOIStartRequest


ACTIVE_ASSOCIATION_STATUSES = frozenset(
    {
        "START_RECEIVED",
        "COLLECTING",
        "READY",
        "VALIDATED",
        "INFERRED",
        "REVIEW_REQUIRED",
    }
)


@dataclass(frozen=True, slots=True)
class HandlerStartEvidence:
    handler_id: str
    handler_session_id: str
    cycle_id: str
    trace_id: str
    station_code: str
    surface: str
    product_id: str
    batch_id: str
    tray_id: str
    slot_index: str

    @classmethod
    def from_request(cls, request: AOIStartRequest) -> "HandlerStartEvidence":
        return cls(**request.model_dump())

    def identity_hash(self) -> str:
        identity = {
            key: value
            for key, value in asdict(self).items()
            if key
            in {
                "handler_id",
                "handler_session_id",
                "cycle_id",
                "trace_id",
                "station_code",
                "surface",
            }
        }
        canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def event_matches_evidence(event: object, evidence: HandlerStartEvidence) -> bool:
    return all(
        getattr(event, event_field) == getattr(evidence, evidence_field)
        for event_field, evidence_field in (
            ("handler_id", "handler_id"),
            ("handler_session_id", "handler_session_id"),
            ("cycle_id", "cycle_id"),
            ("trace_id", "trace_id"),
            ("station", "station_code"),
            ("surface", "surface"),
            ("product_id", "product_id"),
            ("batch_id", "batch_id"),
            ("tray_id", "tray_id"),
            ("slot_index", "slot_index"),
        )
    )
