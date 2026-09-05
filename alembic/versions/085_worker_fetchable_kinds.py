"""Artifact kinds a worker can fetch on demand

Revision ID: 085
Revises: 084
Create Date: 2026-09-05

console#422. The claim gate refuses a segment whose models the polling worker cannot load,
and it needs to know which of those the worker could simply go and GET. The daemon already
downloads LoRAs a pose names but this worker has never seen, so LoRAs must not gate.

Reported by the worker rather than assumed by the API: when the daemon learns to fetch
checkpoints (console#423) it starts sending "checkpoint" here and the gate opens with no API
change. NULL means never reported, read as "fetches nothing" — the direction that keeps an
older daemon claiming the work it can already run.
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "085"
down_revision = "084"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("workers", sa.Column("fetchable_kinds", postgresql.JSONB(), nullable=True))


def downgrade() -> None:
    op.drop_column("workers", "fetchable_kinds")
