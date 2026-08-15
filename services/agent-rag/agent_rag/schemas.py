from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Evidence:
    document_id: str
    chunk_id: str
    title: str
    category: str
    text: str
    score: float

    @property
    def citation(self) -> str:
        return f"{self.document_id}#{self.chunk_id}"


@dataclass(frozen=True, slots=True)
class DataQualityAssessment:
    risk_level: str
    findings: list[str]
    evidence_refs: list[Evidence]
    recommended_actions: list[str]


@dataclass(frozen=True, slots=True)
class ReportDraft:
    status: str
    summary: str
    observed_facts: list[str]
    open_questions: list[str]
    similar_incidents: list[str]
    evidence_refs: list[Evidence]


@dataclass(frozen=True, slots=True)
class ReleaseRecommendation:
    action: str
    failed_gates: list[str] = field(default_factory=list)
    evidence_refs: list[Evidence] = field(default_factory=list)
    approval_required: bool = True

