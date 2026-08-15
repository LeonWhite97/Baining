from sqlalchemy import create_engine, event, func, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.models import Attachment, Base, InspectionEvent
from app.services.demo_data import reset_demo


def test_demo_seed_respects_attachment_event_foreign_key() -> None:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(engine, "connect")
    def enable_foreign_keys(dbapi_connection, _connection_record) -> None:
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(engine)
    with Session(engine) as session:
        assert reset_demo(session, 202408) == 24
        assert session.scalar(select(func.count()).select_from(InspectionEvent)) == 24
        assert session.scalar(select(func.count()).select_from(Attachment)) == 96
