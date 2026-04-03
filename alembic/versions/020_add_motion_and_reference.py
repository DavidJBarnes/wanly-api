"""Add motion_keywords and reference_frames to segments

Revision ID: 020
Revises: 019
Create Date: 2026-04-03
"""

import sqlalchemy as sa
from alembic import op

revision = "020"
down_revision = "019"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("segments", sa.Column("motion_keywords", sa.JSON(), nullable=True))
    op.add_column("segments", sa.Column("reference_frames", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("segments", "reference_frames")
    op.drop_column("segments", "motion_keywords")
