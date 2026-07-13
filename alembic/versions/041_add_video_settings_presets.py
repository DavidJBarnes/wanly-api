"""Add video_settings_presets + per-job/segment video_preset_id

Revision ID: 041
Revises: 040
Create Date: 2026-07-12

User-managed named bundles of the 7 sampler params (lightx2v h/l, cfg h/l, steps_total,
high_noise_steps, flow_shift), live-linked from Job (default) and Segment (override). Seeds
the three previously-hardcoded console presets so nothing is lost.
"""
import uuid
from datetime import datetime, timezone

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "041"
down_revision = "040"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "video_settings_presets",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(length=255), nullable=False, unique=True),
        sa.Column("lightx2v_strength_high", sa.Float(), nullable=True),
        sa.Column("lightx2v_strength_low", sa.Float(), nullable=True),
        sa.Column("cfg_high", sa.Float(), nullable=True),
        sa.Column("cfg_low", sa.Float(), nullable=True),
        sa.Column("steps_total", sa.Integer(), nullable=True),
        sa.Column("high_noise_steps", sa.Integer(), nullable=True),
        sa.Column("flow_shift", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column("jobs", sa.Column("video_preset_id", postgresql.UUID(as_uuid=True), nullable=True))
    op.create_foreign_key(
        "fk_jobs_video_preset", "jobs", "video_settings_presets",
        ["video_preset_id"], ["id"], ondelete="SET NULL",
    )
    op.add_column("segments", sa.Column("video_preset_id", postgresql.UUID(as_uuid=True), nullable=True))
    op.create_foreign_key(
        "fk_segments_video_preset", "segments", "video_settings_presets",
        ["video_preset_id"], ["id"], ondelete="SET NULL",
    )

    now = datetime.now(timezone.utc)
    presets = sa.table(
        "video_settings_presets",
        sa.column("id"), sa.column("name"),
        sa.column("lightx2v_strength_high"), sa.column("lightx2v_strength_low"),
        sa.column("cfg_high"), sa.column("cfg_low"),
        sa.column("steps_total"), sa.column("high_noise_steps"), sa.column("flow_shift"),
        sa.column("created_at"), sa.column("updated_at"),
    )
    op.bulk_insert(presets, [
        {"id": uuid.uuid4(), "name": "Lightning", "lightx2v_strength_high": 1, "lightx2v_strength_low": 1,
         "cfg_high": 1, "cfg_low": 1, "steps_total": 4, "high_noise_steps": 2, "flow_shift": 5,
         "created_at": now, "updated_at": now},
        {"id": uuid.uuid4(), "name": "Prompt-Aware", "lightx2v_strength_high": 0, "lightx2v_strength_low": 1,
         "cfg_high": 2.75, "cfg_low": 1, "steps_total": 12, "high_noise_steps": 8, "flow_shift": 5,
         "created_at": now, "updated_at": now},
        {"id": uuid.uuid4(), "name": "High Motion", "lightx2v_strength_high": 0, "lightx2v_strength_low": 1,
         "cfg_high": 3.5, "cfg_low": 1, "steps_total": 12, "high_noise_steps": 8, "flow_shift": 5,
         "created_at": now, "updated_at": now},
    ])


def downgrade() -> None:
    op.drop_constraint("fk_segments_video_preset", "segments", type_="foreignkey")
    op.drop_column("segments", "video_preset_id")
    op.drop_constraint("fk_jobs_video_preset", "jobs", type_="foreignkey")
    op.drop_column("jobs", "video_preset_id")
    op.drop_table("video_settings_presets")
