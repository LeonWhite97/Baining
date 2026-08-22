from __future__ import annotations

import hashlib
from typing import Any

from agent_rag.embeddings import stable_embedding
from agent_rag.retrieval import KnowledgeChunk, seed_chunks
from agent_rag.schemas import Evidence


class MilvusKnowledgeRepository:
    def __init__(self, uri: str, collection_name: str = "knowledge_chunks", dimension: int = 64) -> None:
        try:
            from pymilvus import MilvusClient
        except ImportError as exc:
            raise RuntimeError(
                "Milvus retrieval requires pymilvus and Milvus Lite. "
                "Install the local-milvus extra or place the verified packages on PYTHONPATH."
            ) from exc

        self.client = MilvusClient(uri=uri)
        self.collection_name = collection_name
        self.dimension = dimension

    def initialize(self) -> None:
        records = [self._record_from_chunk(chunk) for chunk in seed_chunks()]
        if not self.client.has_collection(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                dimension=self.dimension,
                metric_type="COSINE",
            )

        if hasattr(self.client, "upsert"):
            self.client.upsert(collection_name=self.collection_name, data=records)
            return

        for record in records:
            try:
                self.client.delete(collection_name=self.collection_name, ids=[record["id"]])
            except Exception:
                pass
        self.client.insert(collection_name=self.collection_name, data=records)

    def retrieve(self, query: str, categories: list[str] | None = None, limit: int = 5) -> list[Evidence]:
        allowed = set(categories or [])
        search_limit = max(limit, 50 if allowed else limit)
        results = self.client.search(
            collection_name=self.collection_name,
            data=[stable_embedding(query, self.dimension)],
            limit=search_limit,
            output_fields=["document_id", "chunk_id", "title", "text", "category"],
        )

        evidence: list[Evidence] = []
        for hit in results[0] if results else []:
            entity = self._entity_from_hit(hit)
            category = str(entity["category"])
            if allowed and category not in allowed:
                continue
            evidence.append(
                Evidence(
                    document_id=str(entity["document_id"]),
                    chunk_id=str(entity["chunk_id"]),
                    title=str(entity["title"]),
                    category=category,
                    text=str(entity["text"]),
                    score=self._score_from_hit(hit),
                )
            )
            if len(evidence) >= limit:
                break
        return evidence

    def _record_from_chunk(self, chunk: KnowledgeChunk) -> dict[str, Any]:
        text = chunk.title + " " + chunk.text
        return {
            "id": self._primary_key(chunk),
            "vector": stable_embedding(text, self.dimension),
            "document_id": chunk.document_id,
            "chunk_id": chunk.chunk_id,
            "title": chunk.title,
            "text": chunk.text,
            "category": chunk.category,
        }

    @staticmethod
    def _primary_key(chunk: KnowledgeChunk) -> int:
        digest = hashlib.sha256(f"{chunk.document_id}#{chunk.chunk_id}".encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)

    @staticmethod
    def _entity_from_hit(hit: Any) -> dict[str, Any]:
        if isinstance(hit, dict):
            entity = hit.get("entity", hit)
        else:
            entity = getattr(hit, "entity", hit)
        if hasattr(entity, "to_dict"):
            entity = entity.to_dict()
        return dict(entity)

    @staticmethod
    def _score_from_hit(hit: Any) -> float:
        if isinstance(hit, dict):
            value = hit.get("distance", hit.get("score", 0.0))
        else:
            value = getattr(hit, "distance", getattr(hit, "score", 0.0))
        return float(value)
