import os

from fastapi import FastAPI

from agent_rag.api.routes import router
from agent_rag.retrieval import InMemoryKnowledgeRepository


def create_app() -> FastAPI:
    app = FastAPI(title="PIS-IN AOI Agent RAG", version="3.5")
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        from agent_rag.postgres_retrieval import PostgresKnowledgeRepository

        repository = PostgresKnowledgeRepository(database_url)
        repository.initialize()
        app.state.repository = repository
    else:
        app.state.repository = InMemoryKnowledgeRepository.seeded()
    app.include_router(router, prefix="/agent-api/v1")
    return app


app = create_app()
