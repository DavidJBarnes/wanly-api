"""Add loras to video_settings_presets

Revision ID: 042
Revises: 041
Create Date: 2026-07-12

A video preset becomes a complete recipe: 1:N LoRAs (each {lora_id, high_weight, low_weight})
alongside the sampler params. Resolved live at claim time when linked.
"""
import sqlalchemy as sa
from alembic import op

revision = "042"
down_revision = "041"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("video_settings_presets", sa.Column("loras", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("video_settings_presets", "loras")
