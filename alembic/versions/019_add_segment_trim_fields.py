"""Add trim_start_frames and trim_end_frames to segments

Revision ID: 019
Revises: 018
Create Date: 2026-03-16
"""

import sqlalchemy as sa
from alembic import op

revision = "019"
down_revision = "018"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("segments", sa.Column("trim_start_frames", sa.Integer(), nullable=False, server_default="0"))
    op.add_column("segments", sa.Column("trim_end_frames", sa.Integer(), nullable=False, server_default="0"))


def downgrade() -> None:
    op.drop_column("segments", "trim_end_frames")
    op.drop_column("segments", "trim_start_frames")
