"""Add prompt to video_settings_presets + drop prompt_presets

Revision ID: 044
Revises: 043
Create Date: 2026-07-13

Retires the standalone Prompt Presets feature: the default prompt now lives on the video
preset (a snapshot default filled at job creation, overridable at submit). Drops the old
prompt_presets table.
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSON, UUID

revision = "044"
down_revision = "043"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("video_settings_presets", sa.Column("prompt", sa.Text(), nullable=True))
    op.drop_table("prompt_presets")


def downgrade() -> None:
    op.create_table(
        "prompt_presets",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(length=255), nullable=False, unique=True),
        sa.Column("prompt", sa.Text(), nullable=False),
        sa.Column("loras", JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.drop_column("video_settings_presets", "prompt")
