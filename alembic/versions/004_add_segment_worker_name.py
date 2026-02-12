"""Add segment worker_name column

Revision ID: 004
Revises: 003
Create Date: 2026-02-12
"""

from alembic import op
import sqlalchemy as sa

revision = "004"
down_revision = "003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("segments", sa.Column("worker_name", sa.String(255), nullable=True))


def downgrade() -> None:
    op.drop_column("segments", "worker_name")
