"""Add notes to video_settings_presets

Revision ID: 048
Revises: 047
Create Date: 2026-07-15

Free-form notes on a video preset (what it's for, gotchas). Not used by generation.
"""
import sqlalchemy as sa
from alembic import op

revision = "048"
down_revision = "047"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("video_settings_presets", sa.Column("notes", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("video_settings_presets", "notes")
