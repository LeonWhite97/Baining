import os
from pathlib import Path

from fastapi import FastAPI

from agent_rag.api.routes import router
from agent_rag.retrieval import InMemoryKnowledgeRepository


def build_repository():
    milvus_db_path = os.getenv("MILVUS_DB_PATH")
    milvus_uri = os.getenv("MILVUS_URI")
    database_url = os.getenv("DATABASE_URL")

    if milvus_db_path or milvus_uri:
        from agent_rag.milvus_retrieval import MilvusKnowledgeRepository

        uri = milvus_uri or str(Path(milvus_db_path).expanduser().resolve())
        if milvus_db_path:
            Path(uri).parent.mkdir(parents=True, exist_ok=True)
        return MilvusKnowledgeRepository(
            uri=uri,
            collection_name=os.getenv("MILVUS_COLLECTION", "knowledge_chunks"),
        )

    if database_url:
        from agent_rag.postgres_retrieval import PostgresKnowledgeRepository

        return PostgresKnowledgeRepository(database_url)

    return InMemoryKnowledgeRepository.seeded()


def create_app() -> FastAPI:
    app = FastAPI(title="PIS-IN AOI Agent RAG", version="3.5")
    repository = build_repository()
    if hasattr(repository, "initialize"):
        repository.initialize()
    app.state.repository = repository
    app.include_router(router, prefix="/agent-api/v1")
    return app


app = create_app()
