"""One-time dataset unlock: datasets.unlocked_at (#363)

Revision ID: 106
Revises: 105
Create Date: 2026-09-29

A set can be reused for training any number of times but cannot change once used (#356,
#358). Now and then one needs a deliberate one-off change -- a wrong face removed, captions
added, an obsolete pool deleted. POST /datasets/{id}/unlock sets this column and clears the
hand lock, and from then on the training lock only counts runs CREATED AFTER it: the runs
that already trained on the set stop locking it, and the next one locks it again.

  datasets.unlocked_at  when the set was last unlocked; NULL = never, every run counts

Nullable, no backfill: every existing set stays exactly as locked as it was. The job rows
are untouched -- each run snapshotted its images and captions when it was created.
"""
import sqlalchemy as sa

from alembic import op

revision = "106"
down_revision = "105"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("datasets", sa.Column("unlocked_at", sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    op.drop_column("datasets", "unlocked_at")
