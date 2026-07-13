"""Add sampler_name + scheduler to video_settings_presets

Revision ID: 043
Revises: 042
Create Date: 2026-07-12

Optional sampler algorithm + scheduler per preset (NULL -> daemon default euler/simple).
Makes the preset a truly complete recipe.
"""
import sqlalchemy as sa
from alembic import op

revision = "043"
down_revision = "042"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("video_settings_presets", sa.Column("sampler_name", sa.String(length=40), nullable=True))
    op.add_column("video_settings_presets", sa.Column("scheduler", sa.String(length=40), nullable=True))


def downgrade() -> None:
    op.drop_column("video_settings_presets", "scheduler")
    op.drop_column("video_settings_presets", "sampler_name")
