from dataclasses import dataclass
from typing import Protocol

from agent_rag.embeddings import cosine_similarity, stable_embedding
from agent_rag.schemas import Evidence


@dataclass(frozen=True, slots=True)
class KnowledgeChunk:
    document_id: str
    chunk_id: str
    title: str
    text: str
    category: str
    metadata: dict[str, object]


class KnowledgeRepository(Protocol):
    def retrieve(self, query: str, categories: list[str] | None = None, limit: int = 5) -> list[Evidence]: ...


class InMemoryKnowledgeRepository:
    def __init__(self, chunks: list[KnowledgeChunk] | None = None) -> None:
        self.chunks = chunks or []

    @classmethod
    def seeded(cls) -> "InMemoryKnowledgeRepository":
        return cls(
            [
                KnowledgeChunk("SOP-3D-001", "01", "球高超限处置SOP", "球高超限时复核3D量测、检查焊膏与回流参数。", "SOP", {}),
                KnowledgeChunk("DEFECT-001", "01", "缺陷字典", "BALL_BRIDGE 表示焊球桥连，需检查间距和印刷偏移。", "DEFECT_DICTIONARY", {}),
                KnowledgeChunk("INC-2024-07", "01", "ST-02历史异常", "ST-02曾出现球桥连，处理措施为清洁钢网并复核首件。", "INCIDENT_REPORT", {"station": "ST-02"}),
                KnowledgeChunk("MODEL-REL-003", "01", "模型发布门禁", "静默错配率必须为零，盲测和回滚演练通过后才可申请发布。", "MODEL_RELEASE", {}),
            ]
        )

    def retrieve(self, query: str, categories: list[str] | None = None, limit: int = 5) -> list[Evidence]:
        query_vector = stable_embedding(query)
        allowed = set(categories or [])
        candidates = [chunk for chunk in self.chunks if not allowed or chunk.category in allowed]
        ranked = sorted(
            candidates,
            key=lambda chunk: cosine_similarity(query_vector, stable_embedding(chunk.title + " " + chunk.text)),
            reverse=True,
        )[:limit]
        return [
            Evidence(chunk.document_id, chunk.chunk_id, chunk.title, chunk.category, chunk.text,
                     cosine_similarity(query_vector, stable_embedding(chunk.title + " " + chunk.text)))
            for chunk in ranked
        ]


def seed_chunks() -> list[KnowledgeChunk]:
    return InMemoryKnowledgeRepository.seeded().chunks
