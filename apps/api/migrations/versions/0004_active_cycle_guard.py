"""Enforce one active cycle per Handler station.

Revision ID: 0004_active_cycle_guard
"""

import sqlalchemy as sa
from alembic import op


revision = "0004_active_cycle_guard"
down_revision = "0003_capture_identity"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("inspection_events", sa.Column("active_cycle_guard", sa.String(length=8), nullable=True))
    op.create_unique_constraint(
        "uq_handler_station_active",
        "inspection_events",
        ["handler_id", "station", "active_cycle_guard"],
    )


def downgrade() -> None:
    op.drop_constraint("uq_handler_station_active", "inspection_events", type_="unique")
    op.drop_column("inspection_events", "active_cycle_guard")
