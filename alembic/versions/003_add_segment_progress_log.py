"""Add segment progress_log column

Revision ID: 003
Revises: 002
Create Date: 2026-02-12
"""

from alembic import op
import sqlalchemy as sa

revision = "003"
down_revision = "002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("segments", sa.Column("progress_log", sa.Text, nullable=True))


def downgrade() -> None:
    op.drop_column("segments", "progress_log")
