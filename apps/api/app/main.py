import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import func, select

from app.api.routes.health import router as health_router
from app.api.routes.handler import router as handler_router
from app.api.routes.operations import router as operations_router
from app.clients.agent_rag import AgentRagClient
from app.config import InferenceSettings, RuntimeSettings
from app.db import create_session_factory
from app.models import Base
from app.models import InspectionEvent
from app.inference.base import InferenceAdapter
from app.inference.factory import build_inference_adapter
from app.services.demo_data import reset_demo


def create_app(
    *,
    database_url: str | None = None,
    mode: str | None = None,
    auto_seed: int | None = None,
    auto_pass_enabled: bool | str | None = None,
    handler_integration_enabled: bool | str | None = None,
    image_root: str | Path | None = None,
    inference_adapter: InferenceAdapter | None = None,
) -> FastAPI:
    app = FastAPI(title="PIS-IN AOI AI", version="3.5")
    settings = RuntimeSettings.from_values(
        mode=mode,
        auto_pass_enabled=auto_pass_enabled,
        handler_integration_enabled=handler_integration_enabled,
    )
    app.state.database_url = database_url or os.getenv("DATABASE_URL", "sqlite+pysqlite:///./aoi_demo.db")
    app.state.settings = settings
    app.state.mode = settings.mode.value
    app.state.auto_pass_enabled = settings.auto_pass_enabled
    app.state.handler_integration_enabled = settings.handler_integration_enabled
    inference_settings = InferenceSettings.from_values()
    app.state.inference_settings = inference_settings
    app.state.inference_adapter = inference_adapter or build_inference_adapter(
        settings.mode, inference_settings
    )
    configured_image_root = image_root or os.getenv("AOI_IMAGE_ROOT", "./aoi-images-disabled")
    app.state.image_root = Path(configured_image_root).resolve()
    app.state.session_factory = create_session_factory(app.state.database_url)
    agent_rag_url = os.getenv("AGENT_RAG_URL")
    app.state.agent_client = (
        AgentRagClient(agent_rag_url)
        if app.state.mode != "demo" and agent_rag_url
        else None
    )
    Base.metadata.create_all(app.state.session_factory.kw["bind"])
    configured_seed = auto_seed or int(os.getenv("DEMO_AUTO_SEED", "0")) or None
    if app.state.mode == "demo" and configured_seed is not None:
        with app.state.session_factory() as session:
            event_count = session.scalar(select(func.count()).select_from(InspectionEvent)) or 0
            if event_count == 0:
                reset_demo(session, configured_seed)
    app.include_router(health_router, prefix="/api/v1")
    app.include_router(operations_router, prefix="/api/v1")
    if app.state.handler_integration_enabled and app.state.mode in {"shadow", "controlled"}:
        app.include_router(handler_router, prefix="/api/v1")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://localhost:5174", "http://127.0.0.1:5173", "http://127.0.0.1:5174"],
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type"],
    )
    return app


app = create_app()
