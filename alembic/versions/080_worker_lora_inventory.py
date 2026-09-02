"""What LoRAs a worker holds, as of its last sync

Revision ID: 080
Revises: 079
Create Date: 2026-09-02

A worker is the only thing that can see its own LoRA directory, so the console cannot diff
for it — it has to be told. Reported through the heartbeat and stored here, following
gpu_stats / sd_scripts / a1111, which are the same shape for the same reason.

NULLABLE: a worker that has not reported yet, or one running an older daemon, has no
inventory rather than an empty one. Those are different — "nothing here" and "not asked"
should not render identically on a status page.

The payload is a CACHE of the last sync's verdict, carrying `synced_at`, never a live
check. Verifying a LoRA means hashing 650 MB; a status page cannot ask for that on every
heartbeat, and pretending it is live would be a lie with a timestamp missing.
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "080"
down_revision = "079"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("workers", sa.Column("loras", postgresql.JSONB(), nullable=True))


def downgrade() -> None:
    op.drop_column("workers", "loras")
