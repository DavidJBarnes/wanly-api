"""Add tags column to videos

Revision ID: 025
Revises: 024
Create Date: 2026-05-07
"""

from alembic import op
import sqlalchemy as sa

revision = "025"
down_revision = "024"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "videos",
        sa.Column("tags", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("videos", "tags")
