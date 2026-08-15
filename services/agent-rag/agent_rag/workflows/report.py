from agent_rag.retrieval import InMemoryKnowledgeRepository, KnowledgeRepository
from agent_rag.schemas import ReportDraft


def run_report_workflow(payload: dict[str, object], repository: KnowledgeRepository | None = None) -> ReportDraft:
    repository = repository or InMemoryKnowledgeRepository.seeded()
    station = str(payload.get("station", "UNKNOWN"))
    defect_code = str(payload.get("defect_code") or "UNKNOWN")
    rate = float(payload.get("defect_rate", 0.0))
    threshold = float(payload.get("threshold", 0.0))
    sample_count = int(payload.get("sample_count", 0))
    facts = [
        f"工站 {station} 缺陷率 {rate:.1%}，阈值 {threshold:.1%}",
        f"统计窗口样本量 {sample_count}",
        f"主要缺陷类别 {defect_code}",
    ]
    evidence = repository.retrieve(f"{station} {defect_code} 异常处置", ["INCIDENT_REPORT", "DEFECT_DICTIONARY"], 3)
    questions = ["是否发生换线或参数调整？", "是否存在同批次物料集中异常？"]
    return ReportDraft(
        status="DRAFT",
        summary=f"{station} 缺陷率超过阈值，建议由质量与工艺人员联合确认。",
        observed_facts=facts,
        open_questions=questions,
        similar_incidents=[item.citation for item in evidence],
        evidence_refs=evidence,
    )
