"""Base models each worker can load

Revision ID: 082
Revises: 081
Create Date: 2026-09-03

Reported through the heartbeat, following loras (080) for the same reason: a worker is the
only thing that can see this. The engine binds to 127.0.0.1 inside the container, so nothing
upstream can ask it what ComfyUI will load — and "what ComfyUI will load" is the real
question, since a checkpoint the folder mapping does not cover is invisible to a render
however present it is on disk.

NULLABLE: never reported is not the same as none available, and a console dropdown should
not present those identically.
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "082"
down_revision = "081"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("workers", sa.Column("checkpoints", postgresql.JSONB(), nullable=True))


def downgrade() -> None:
    op.drop_column("workers", "checkpoints")
