"""Character-LoRA training as claimable work

Revision ID: 090
Revises: 089
Create Date: 2026-09-07

wanly-api#274, layer 1 of wanly-console#453. Training a LoRA has been a laptop-driven ssh
pipeline; this makes it a row the console creates and a worker claims.

SHAPED ON `segments`, deliberately. It is the same problem -- a row created here, claimed by a
remote worker, worked on for a long time, reported back about -- and the machinery that already
exists for that (orphan reclaim, the heartbeat sweep, the console's status rendering) only
transfers if the shape does. Hence status/worker_id/worker_name/gpu_name/claimed_at/
completed_at/error_message/progress_log carrying exactly their `segments` meanings.

A separate table rather than a variant of `segments` because almost nothing overlaps: no job, no
index, no seed, no video. They share a lifecycle, not a payload.

Two columns `segments` has no equivalent of:

  step / total_steps   a training run genuinely has a step count, and the console already has a
                       determinate progress bar looking for exactly these two numbers. Renders
                       have no such thing, which is why Segment reports progress as prose.

  checkpoints          every epoch is a candidate and choosing between them is a human
                       judgement made by eye at a fixed seed. Loss does NOT rank them -- a
                       confident "later epochs overfit" call read off a loss curve was refuted
                       outright on d0ggyff -- so all of them are recorded, not just the last.

THE PARTIAL UNIQUE INDEX is the interesting one. One LIVE run per (character, version): a second
attempt at v2 while the first is still going would train two LoRAs into the same output name and
the second would win silently. That is the same collision new_character.sh used to produce by
reusing one run directory per character, and it was invisible until someone read file mtimes.
Terminal rows are excluded so a failed v2 can simply be retried.
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "090"
down_revision = "089"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "training_jobs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("users.id"), nullable=True),
        sa.Column("character", sa.String(length=64), nullable=False),
        sa.Column("trigger", sa.String(length=64), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("dataset_images", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("config", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="pending"),
        sa.Column("worker_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("worker_name", sa.String(length=255), nullable=True),
        sa.Column("gpu_name", sa.String(length=100), nullable=True),
        sa.Column("progress_log", sa.Text(), nullable=True),
        sa.Column("step", sa.Integer(), nullable=True),
        sa.Column("total_steps", sa.Integer(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("checkpoints", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("output_lora_path", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("claimed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_training_jobs_status", "training_jobs", ["status"])
    op.create_index(
        "uq_training_jobs_character_version_live", "training_jobs", ["character", "version"],
        unique=True,
        postgresql_where=sa.text("status NOT IN ('completed','failed','cancelled')"),
    )


def downgrade() -> None:
    op.drop_index("uq_training_jobs_character_version_live", table_name="training_jobs")
    op.drop_index("ix_training_jobs_status", table_name="training_jobs")
    op.drop_table("training_jobs")
