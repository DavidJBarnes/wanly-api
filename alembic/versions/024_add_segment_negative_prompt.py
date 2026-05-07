"""Add negative_prompt column to segments

Revision ID: 024
Revises: 023
Create Date: 2026-05-07
"""

from alembic import op
import sqlalchemy as sa

revision = "024"
down_revision = "023"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "segments",
        sa.Column("negative_prompt", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("segments", "negative_prompt")
