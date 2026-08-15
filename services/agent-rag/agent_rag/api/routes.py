from dataclasses import asdict

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from agent_rag.workflows.data_quality import run_data_quality_workflow
from agent_rag.workflows.model_governance import evaluate_release
from agent_rag.workflows.report import run_report_workflow


router = APIRouter()


class SearchIn(BaseModel):
    query: str = Field(min_length=1)
    categories: list[str] = Field(default_factory=list)
    limit: int = Field(default=5, ge=1, le=20)


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "provider": "deterministic-demo"}


@router.post("/knowledge/search")
def search(payload: SearchIn, request: Request) -> dict[str, object]:
    items = request.app.state.repository.retrieve(payload.query, payload.categories, payload.limit)
    return {"items": [{**asdict(item), "citation": item.citation} for item in items]}


@router.post("/assess-data-quality")
def assess_data_quality(payload: dict[str, object], request: Request) -> dict[str, object]:
    return asdict(run_data_quality_workflow(payload, request.app.state.repository))


@router.post("/draft-report")
def draft_report(payload: dict[str, object], request: Request) -> dict[str, object]:
    return asdict(run_report_workflow(payload, request.app.state.repository))


@router.post("/recommend-model-release")
def recommend_model_release(payload: dict[str, object]) -> dict[str, object]:
    return asdict(evaluate_release(payload))
