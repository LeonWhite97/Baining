from collections.abc import Iterator

from fastapi import Request
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool


def create_session_factory(database_url: str) -> sessionmaker[Session]:
    engine_options: dict[str, object] = {"pool_pre_ping": True}
    if database_url.startswith("sqlite"):
        engine_options["connect_args"] = {"check_same_thread": False}
        if database_url.endswith(":memory:"):
            engine_options["poolclass"] = StaticPool
    engine = create_engine(database_url, **engine_options)
    return sessionmaker(bind=engine, expire_on_commit=False)


def get_session(request: Request) -> Iterator[Session]:
    with request.app.state.session_factory() as session:
        session.info["image_root"] = request.app.state.image_root
        session.info["mode"] = request.app.state.mode
        yield session
