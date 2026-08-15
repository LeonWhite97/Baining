"""Add Handler-AOI safety foundation.

Revision ID: 0002_handler_aoi_safety
"""

import sqlalchemy as sa
from alembic import op


revision = "0002_handler_aoi_safety"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("inspection_events", sa.Column("trace_id", sa.String(length=128), nullable=True))
    op.add_column("inspection_events", sa.Column("handler_id", sa.String(length=64), nullable=True))
    op.add_column("inspection_events", sa.Column("handler_session_id", sa.String(length=64), nullable=True))
    op.add_column("inspection_events", sa.Column("cycle_id", sa.String(length=64), nullable=True))
    op.add_column("inspection_events", sa.Column("capture_id", sa.String(length=36), nullable=True))
    op.add_column("inspection_events", sa.Column("camera_trigger_sequence", sa.BigInteger(), nullable=True))
    op.add_column("inspection_events", sa.Column("capture_started_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("inspection_events", sa.Column("capture_completed_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("inspection_events", sa.Column("cycle_deadline_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("inspection_events", sa.Column("aoi_bin", sa.Integer(), nullable=True))
    op.add_column("inspection_events", sa.Column("result_category", sa.String(length=16), nullable=True))
    op.add_column("inspection_events", sa.Column("human_decision", sa.String(length=16), nullable=True))
    op.add_column("inspection_events", sa.Column("final_decision", sa.String(length=16), nullable=True))
    op.add_column("inspection_events", sa.Column("decision_source", sa.String(length=16), nullable=True))
    op.add_column(
        "inspection_events",
        sa.Column("handler_publish_status", sa.String(length=32), nullable=False, server_default="NOT_READY"),
    )
    op.add_column(
        "inspection_events",
        sa.Column("mes_publish_status", sa.String(length=32), nullable=False, server_default="NOT_READY"),
    )
    op.add_column(
        "inspection_events",
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.alter_column("inspection_events", "ai_decision", existing_type=sa.String(length=16), nullable=True)
    op.alter_column("inspection_events", "ai_confidence", existing_type=sa.Float(), nullable=True)
    op.alter_column("inspection_events", "reason_code", existing_type=sa.String(length=64), nullable=True)
    op.alter_column("inspection_events", "image_url", existing_type=sa.String(length=256), nullable=True)
    op.create_index("ix_inspection_events_trace_id", "inspection_events", ["trace_id"], unique=True)
    op.create_index("ix_inspection_events_handler_id", "inspection_events", ["handler_id"], unique=False)
    op.create_index("uq_handler_cycle", "inspection_events", ["handler_id", "handler_session_id", "cycle_id"], unique=True)
    op.create_index("uq_inspection_capture_id", "inspection_events", ["capture_id"], unique=True)

    op.create_table(
        "handler_result_outbox",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("message_id", sa.String(length=64), nullable=False, unique=True),
        sa.Column("event_uuid", sa.String(length=64), sa.ForeignKey("inspection_events.event_uuid", ondelete="CASCADE"), nullable=False),
        sa.Column("trace_id", sa.String(length=128), nullable=False),
        sa.Column("cycle_id", sa.String(length=64), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="PENDING"),
        sa.Column("retry_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("next_retry_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("delivery_uncertain_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_error", sa.String(length=512), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_handler_result_outbox_status", "handler_result_outbox", ["status"])
    op.create_index("ix_handler_result_outbox_trace_id", "handler_result_outbox", ["trace_id"])

    op.create_table(
        "mes_outbox",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("message_id", sa.String(length=64), nullable=False, unique=True),
        sa.Column("event_uuid", sa.String(length=64), sa.ForeignKey("inspection_events.event_uuid", ondelete="CASCADE"), nullable=False),
        sa.Column("event_revision", sa.Integer(), nullable=False),
        sa.Column("event_type", sa.String(length=32), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="PENDING"),
        sa.Column("retry_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("next_retry_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_error", sa.String(length=512), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("event_uuid", "event_revision", name="uq_mes_event_revision"),
    )
    op.create_index("ix_mes_outbox_status", "mes_outbox", ["status"])


def downgrade() -> None:
    op.drop_index("ix_mes_outbox_status", table_name="mes_outbox")
    op.drop_table("mes_outbox")
    op.drop_index("ix_handler_result_outbox_trace_id", table_name="handler_result_outbox")
    op.drop_index("ix_handler_result_outbox_status", table_name="handler_result_outbox")
    op.drop_table("handler_result_outbox")
    op.drop_index("uq_inspection_capture_id", table_name="inspection_events")
    op.drop_index("uq_handler_cycle", table_name="inspection_events")
    op.drop_index("ix_inspection_events_handler_id", table_name="inspection_events")
    op.drop_index("ix_inspection_events_trace_id", table_name="inspection_events")
    for column in (
        "updated_at", "mes_publish_status", "handler_publish_status", "decision_source", "final_decision",
        "human_decision", "result_category", "aoi_bin", "cycle_deadline_at", "capture_completed_at",
        "capture_started_at", "camera_trigger_sequence", "capture_id", "cycle_id", "handler_session_id",
        "handler_id", "trace_id",
    ):
        op.drop_column("inspection_events", column)
    op.alter_column("inspection_events", "image_url", existing_type=sa.String(length=256), nullable=False)
    op.alter_column("inspection_events", "reason_code", existing_type=sa.String(length=64), nullable=False)
    op.alter_column("inspection_events", "ai_confidence", existing_type=sa.Float(), nullable=False)
    op.alter_column("inspection_events", "ai_decision", existing_type=sa.String(length=16), nullable=False)
