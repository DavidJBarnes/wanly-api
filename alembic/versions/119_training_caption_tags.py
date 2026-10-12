"""training_jobs.caption_tags: the WD14 tags that landed in an SDXL run's captions (wanly-api#462)

{images, groups: [{label, kind, images, kept: [[tag, n]], stripped: [[tag, n]] | null}]}, as
the trainer reports it after staging. A tag the caption names is learned onto that tag, not
the trigger, so this is what shows which identity features the trigger did NOT learn.
`stripped` is null on backfilled runs: a stripped tag leaves no trace in a caption file.

Revision ID: 119
Revises: 118
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "119"
down_revision = "118"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("training_jobs", sa.Column("caption_tags", postgresql.JSONB(), nullable=True))


def downgrade() -> None:
    op.drop_column("training_jobs", "caption_tags")
