"""workers.phase: what a worker is doing right now (wanly-gpu-daemon#189)

A busy worker must never look offline. The daemon now registers before its LoRA sync and
reports a `phase` on every heartbeat -- syncing_loras with file / MB / rate / ETA, or the
supervisor's booting / switching_mode -- and the console shows it in place of the status.
NULL means never reported (an older daemon), not "idle".

Revision ID: 121
Revises: 120
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "121"
down_revision = "120"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("workers", sa.Column("phase", postgresql.JSONB(), nullable=True))


def downgrade() -> None:
    op.drop_column("workers", "phase")
