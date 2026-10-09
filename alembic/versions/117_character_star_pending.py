"""ltx_characters.star_pending: a starred checkpoint still on the trainer (wanly-api#452)

The character's LoRA is the user's STAR, not whichever run finished last. Starring a checkpoint
that is not in the bucket yet (publish "none" is the default, #413) asks for its upload and
records the star here; it applies when the file lands.

Revision ID: 117
Revises: 116
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "117"
down_revision = "116"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("ltx_characters", sa.Column("star_pending", postgresql.JSONB(), nullable=True))


def downgrade() -> None:
    op.drop_column("ltx_characters", "star_pending")
