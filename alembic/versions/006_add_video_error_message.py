"""Add error_message column to videos table

Revision ID: 006
Revises: 005
Create Date: 2026-02-13
"""

from alembic import op
import sqlalchemy as sa

revision = "006"
down_revision = "005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("videos", sa.Column("error_message", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("videos", "error_message")
