"""Add lightx2v strength columns to jobs

Revision ID: 012
Revises: 011
Create Date: 2026-02-26
"""

from alembic import op
import sqlalchemy as sa

revision = "012"
down_revision = "011"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "jobs",
        sa.Column("lightx2v_strength_high", sa.Float(), nullable=True),
    )
    op.add_column(
        "jobs",
        sa.Column("lightx2v_strength_low", sa.Float(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("jobs", "lightx2v_strength_low")
    op.drop_column("jobs", "lightx2v_strength_high")
