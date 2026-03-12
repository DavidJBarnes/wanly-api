"""Add cfg_high and cfg_low columns to jobs

Revision ID: 016
Revises: 015
Create Date: 2026-03-12
"""

from alembic import op
import sqlalchemy as sa

revision = "016"
down_revision = "015"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "jobs",
        sa.Column("cfg_high", sa.Float(), nullable=True),
    )
    op.add_column(
        "jobs",
        sa.Column("cfg_low", sa.Float(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("jobs", "cfg_low")
    op.drop_column("jobs", "cfg_high")
