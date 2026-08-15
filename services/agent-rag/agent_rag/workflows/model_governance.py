from agent_rag.retrieval import InMemoryKnowledgeRepository
from agent_rag.schemas import ReleaseRecommendation


def evaluate_release(metrics: dict[str, object]) -> ReleaseRecommendation:
    failed: list[str] = []
    if float(metrics.get("silent_mismatch_rate", 0)) > 0:
        failed.append("SILENT_MISMATCH")
    if not bool(metrics.get("blind_test_passed")):
        failed.append("BLIND_TEST")
    if not bool(metrics.get("rollback_drill_passed")):
        failed.append("ROLLBACK_DRILL")
    if bool(metrics.get("critical_defect_escape")):
        failed.append("CRITICAL_DEFECT_ESCAPE")
    if float(metrics.get("backlog_growth", 0)) > 0:
        failed.append("BACKLOG_GROWTH")
    if not bool(metrics.get("approval_metadata")):
        failed.append("APPROVAL_METADATA")
    action = "BLOCK" if failed else "READY_FOR_APPROVAL"
    evidence = InMemoryKnowledgeRepository.seeded().retrieve("模型发布门禁", ["MODEL_RELEASE"], 2)
    return ReleaseRecommendation(action, failed, evidence, approval_required=True)
