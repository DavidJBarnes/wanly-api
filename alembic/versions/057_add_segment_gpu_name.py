"""Record which GPU ran each segment

Revision ID: 057
Revises: 056
Create Date: 2026-08-07

Run times vary enormously by hardware, and there was no way to see it — comparing a 3090 against
a 4090 meant reading log lines by hand.

The GPU name is only knowable from the worker, and worker rows are deleted when a worker
deregisters: every RunPod pod takes its row with it when it drains. So the name has to be
snapshotted onto the segment at claim time, or the association is lost the moment the pod goes
away.

The backfill recovers what it honestly can — segments whose worker row still exists, which is
mostly the long-lived 3090. Segments run by pods that have since terminated stay NULL rather
than being guessed at from a worker name someone typed by hand.
"""
import sqlalchemy as sa
from alembic import op

revision = "057"
down_revision = "056"
branch_labels = None
depends_on = None


# ->> extracts the JSONB value as text; IS NOT NULL is used rather than the `?` containment
# operator because `?` collides with DBAPI parameter binding.
_BACKFILL = sa.text(
    """
    UPDATE segments AS s
    SET gpu_name = w.gpu_stats->>'gpu_name'
    FROM workers AS w
    WHERE s.worker_id = w.id
      AND s.gpu_name IS NULL
      AND w.gpu_stats->>'gpu_name' IS NOT NULL
    """
)


def upgrade() -> None:
    op.add_column("segments", sa.Column("gpu_name", sa.String(length=100), nullable=True))
    op.execute(_BACKFILL)


def downgrade() -> None:
    op.drop_column("segments", "gpu_name")
