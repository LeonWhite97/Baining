from enum import StrEnum


class Decision(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    REVIEW = "REVIEW"


class AssociationStatus(StrEnum):
    RECEIVED = "RECEIVED"
    COLLECTING = "COLLECTING"
    READY = "READY"
    VALIDATED = "VALIDATED"
    INFERRED = "INFERRED"
    ARCHIVED = "ARCHIVED"
    EXPIRED = "EXPIRED"
    QUARANTINED = "QUARANTINED"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    INVALID = "INVALID"

    @property
    def is_terminal(self) -> bool:
        return self in {
            self.ARCHIVED,
            self.EXPIRED,
            self.QUARANTINED,
            self.INVALID,
        }


class InferenceMode(StrEnum):
    DEMO = "DEMO"
    FULL = "FULL"
    TWO_D_ONLY = "2D_ONLY"
    PARTIAL = "PARTIAL"


class AlertStatus(StrEnum):
    OPEN = "OPEN"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    CLOSED = "CLOSED"


class ReportStatus(StrEnum):
    DRAFT = "DRAFT"
    APPROVED = "APPROVED"


class Resolution(StrEnum):
    MANUAL_REVIEW = "MANUAL_REVIEW"
    REPROCESSED = "REPROCESSED"
    DISCARDED = "DISCARDED"


class ResultCategory(StrEnum):
    QUALITY = "QUALITY"
    SYSTEM = "SYSTEM"


class PublishStatus(StrEnum):
    NOT_READY = "NOT_READY"
    PENDING = "PENDING"
    SENDING = "SENDING"
    SENT = "SENT"
    ACKED = "ACKED"
    RETRYING = "RETRYING"
    DELIVERY_UNCERTAIN = "DELIVERY_UNCERTAIN"
    EXPIRED = "EXPIRED"
