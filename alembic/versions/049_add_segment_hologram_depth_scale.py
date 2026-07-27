"""Add hologram_depth_scale_m to segments

Revision ID: 049
Revises: 048
Create Date: 2026-07-27

Per-remake relief depth (meters) for the 2.5d_depth hologram flavor. Was a daemon-wide
config constant (0.12) — now a console dialog knob threaded through the carrier segment.
"""
import sqlalchemy as sa
from alembic import op

revision = "049"
down_revision = "048"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("segments", sa.Column("hologram_depth_scale_m", sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("segments", "hologram_depth_scale_m")
