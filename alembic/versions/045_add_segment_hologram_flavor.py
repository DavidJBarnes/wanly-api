"""Add hologram_flavor to segments

Revision ID: 045
Revises: 044
Create Date: 2026-07-13

AR hologram gains a flavor: "2d_matte" (Tier-0 flat) or "2.5d_depth" (Tier-1 depth-displaced).
"""
import sqlalchemy as sa
from alembic import op

revision = "045"
down_revision = "044"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("segments", sa.Column("hologram_flavor", sa.String(length=16), nullable=True))


def downgrade() -> None:
    op.drop_column("segments", "hologram_flavor")
