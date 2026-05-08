"""Add index on video tags column

Revision ID: 026
Revises: 025
Create Date: 2026-05-08
"""

from alembic import op
import sqlalchemy as sa

revision = "026"
down_revision = "025"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index("ix_videos_tags", "videos", ["tags"])


def downgrade() -> None:
    op.drop_index("ix_videos_tags")
