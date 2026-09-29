"""Manual dataset lock: datasets.locked_at, datasets.locked_reason (#358)

Revision ID: 104
Revises: 103
Create Date: 2026-09-28

#356 locks a set once a run that is not failed or cancelled has trained on it, and derives
that lock from training_jobs on every read. Some sets need locking without having trained --
a set kept as a reference, or one whose LoRA was trained before provenance was recorded --
and nothing in training_jobs can say so. These two columns can:

  datasets.locked_at      when POST /datasets/{id}/lock was called; NULL = not locked by hand
  datasets.locked_reason  what the person locking it said, if anything

One-way: no route clears them, and a clone does not copy them. Both nullable, no backfill:
every existing set stays exactly as locked as it was.
"""
import sqlalchemy as sa

from alembic import op

revision = "104"
down_revision = "103"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("datasets", sa.Column("locked_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("datasets", sa.Column("locked_reason", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("datasets", "locked_reason")
    op.drop_column("datasets", "locked_at")
