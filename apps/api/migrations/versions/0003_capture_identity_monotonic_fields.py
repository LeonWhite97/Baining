"""Persist capture monotonic identity fields.

Revision ID: 0003_capture_identity
"""

import sqlalchemy as sa
from alembic import op


revision = "0003_capture_identity"
down_revision = "0002_handler_aoi_safety"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("inspection_events", sa.Column("start_received_monotonic_ns", sa.BigInteger(), nullable=True))
    op.add_column("inspection_events", sa.Column("capture_started_monotonic_ns", sa.BigInteger(), nullable=True))
    op.add_column("inspection_events", sa.Column("capture_trigger_source", sa.String(length=24), nullable=True))


def downgrade() -> None:
    op.drop_column("inspection_events", "capture_trigger_source")
    op.drop_column("inspection_events", "capture_started_monotonic_ns")
    op.drop_column("inspection_events", "start_received_monotonic_ns")
