"""Add jobs.continuation_mode

Revision ID: 038
Revises: 037
Create Date: 2026-07-11

Per-job override for segment continuation strategy ("traditional" | "vace").
NULL falls back to the global continuation_mode app setting. Part of the VACE
continuation integration (default off).
"""
import sqlalchemy as sa
from alembic import op

revision = "038"
down_revision = "037"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("jobs", sa.Column("continuation_mode", sa.String(length=20), nullable=True))


def downgrade() -> None:
    op.drop_column("jobs", "continuation_mode")
