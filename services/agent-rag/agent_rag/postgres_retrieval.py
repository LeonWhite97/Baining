import json

from sqlalchemy import create_engine, text

from agent_rag.embeddings import stable_embedding
from agent_rag.retrieval import KnowledgeChunk, seed_chunks
from agent_rag.schemas import Evidence


class PostgresKnowledgeRepository:
    def __init__(self, database_url: str) -> None:
        self.engine = create_engine(database_url, pool_pre_ping=True)

    def initialize(self) -> None:
        with self.engine.begin() as connection:
            connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            connection.execute(text("""
                CREATE TABLE IF NOT EXISTS knowledge_chunks (
                    id BIGSERIAL PRIMARY KEY,
                    document_id VARCHAR(64) NOT NULL,
                    chunk_id VARCHAR(64) NOT NULL,
                    title VARCHAR(256) NOT NULL,
                    text TEXT NOT NULL,
                    category VARCHAR(32) NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                    embedding vector(64) NOT NULL,
                    UNIQUE(document_id, chunk_id)
                )
            """))
            for chunk in seed_chunks():
                connection.execute(
                    text("""
                        INSERT INTO knowledge_chunks(document_id, chunk_id, title, text, category, metadata, embedding)
                        VALUES (:document_id, :chunk_id, :title, :text, :category, CAST(:metadata AS jsonb), CAST(:embedding AS vector))
                        ON CONFLICT(document_id, chunk_id) DO NOTHING
                    """),
                    {
                        "document_id": chunk.document_id, "chunk_id": chunk.chunk_id, "title": chunk.title,
                        "text": chunk.text, "category": chunk.category, "metadata": json.dumps(chunk.metadata),
                        "embedding": str(stable_embedding(chunk.title + " " + chunk.text)),
                    },
                )

    def retrieve(self, query: str, categories: list[str] | None = None, limit: int = 5) -> list[Evidence]:
        filters = "AND category = ANY(:categories)" if categories else ""
        statement = text(f"""
            SELECT document_id, chunk_id, title, text, category,
                   1 - (embedding <=> CAST(:embedding AS vector)) AS score
            FROM knowledge_chunks
            WHERE 1=1 {filters}
            ORDER BY embedding <=> CAST(:embedding AS vector)
            LIMIT :limit
        """)
        parameters: dict[str, object] = {"embedding": str(stable_embedding(query)), "limit": limit}
        if categories:
            parameters["categories"] = categories
        with self.engine.connect() as connection:
            rows = connection.execute(statement, parameters).mappings().all()
        return [Evidence(row["document_id"], row["chunk_id"], row["title"], row["category"], row["text"], float(row["score"])) for row in rows]
