from agent_rag.retrieval import InMemoryKnowledgeRepository, KnowledgeRepository
from agent_rag.schemas import DataQualityAssessment


def run_data_quality_workflow(payload: dict[str, object], repository: KnowledgeRepository | None = None) -> DataQualityAssessment:
    repository = repository or InMemoryKnowledgeRepository.seeded()
    findings: list[str] = []
    actions: list[str] = []
    identity_complete = bool(payload.get("identity_complete"))
    attachments = payload.get("attachments") or []
    if not identity_complete:
        findings.append("检测身份字段不完整，无法安全归属事件")
        actions.extend(["QUARANTINE", "MANUAL_REVIEW"])
        return DataQualityAssessment("CRITICAL", findings, repository.retrieve("身份缺失", ["SOP"], 1), actions)
    if not attachments:
        findings.append("未收到预期附件")
        actions.append("WAIT_OR_RETRY")
    if payload.get("duplicate_file"):
        findings.append("发现重复文件哈希")
        actions.append("DEDUPLICATE")
    if payload.get("capture_delta_ms", 0) and abs(int(payload["capture_delta_ms"])) > int(payload.get("capture_delta_threshold_ms", 500)):
        findings.append("采集时间偏差超阈值")
        actions.append("QUARANTINE")
    risk = "HIGH" if findings else "LOW"
    return DataQualityAssessment(risk, findings, repository.retrieve("数据完整性处置", ["SOP"], 2), actions or ["OBSERVE"])
