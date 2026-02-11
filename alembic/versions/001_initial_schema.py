"""Initial schema

Revision ID: 001
Revises:
Create Date: 2026-02-11
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Users
    op.create_table(
        "users",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("username", sa.String(255), unique=True, nullable=False),
        sa.Column("password_hash", sa.String(255), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )

    # Jobs
    op.create_table(
        "jobs",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("user_id", UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("width", sa.Integer, nullable=False),
        sa.Column("height", sa.Integer, nullable=False),
        sa.Column("fps", sa.Integer, nullable=False),
        sa.Column("seed", sa.BigInteger, nullable=False),
        sa.Column("starting_image", sa.Text, nullable=True),
        sa.Column("status", sa.String(20), nullable=False, server_default="pending"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )
    op.create_index("ix_jobs_user_id", "jobs", ["user_id"])
    op.create_index("ix_jobs_status", "jobs", ["status"])

    # Segments
    op.create_table(
        "segments",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("job_id", UUID(as_uuid=True), sa.ForeignKey("jobs.id"), nullable=False),
        sa.Column("index", sa.Integer, nullable=False),
        sa.Column("prompt", sa.Text, nullable=False),
        sa.Column("duration_seconds", sa.Float, nullable=False, server_default="5.0"),
        sa.Column("start_image", sa.Text, nullable=True),
        sa.Column("loras", sa.JSON, nullable=True),
        sa.Column("faceswap_enabled", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("faceswap_method", sa.String(20), nullable=True),
        sa.Column("faceswap_source_type", sa.String(20), nullable=True),
        sa.Column("faceswap_image", sa.Text, nullable=True),
        sa.Column("faceswap_faces_order", sa.Text, nullable=True),
        sa.Column("faceswap_faces_index", sa.Text, nullable=True),
        sa.Column("auto_finalize", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("status", sa.String(20), nullable=False, server_default="pending"),
        sa.Column("worker_id", UUID(as_uuid=True), nullable=True),
        sa.Column("output_path", sa.Text, nullable=True),
        sa.Column("last_frame_path", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("claimed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_message", sa.Text, nullable=True),
    )
    op.create_unique_constraint("uq_segments_job_index", "segments", ["job_id", "index"])
    op.create_index("ix_segments_job_id", "segments", ["job_id"])
    op.create_index("ix_segments_status", "segments", ["status"])

    # Videos
    op.create_table(
        "videos",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("job_id", UUID(as_uuid=True), sa.ForeignKey("jobs.id"), nullable=False),
        sa.Column("output_path", sa.Text, nullable=True),
        sa.Column("duration_seconds", sa.Float, nullable=True),
        sa.Column("status", sa.String(20), nullable=False, server_default="pending"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_videos_job_id", "videos", ["job_id"])

    # Seed user: dbarnes
    op.execute(
        "INSERT INTO users (id, username, password_hash) VALUES ("
        "gen_random_uuid(), 'dbarnes', "
        "'$2b$12$st4zgxhI3j9msjUoufujVuawe3VXav.eZJ/uTCeSEGloG5UliPj6a'"
        ")"
    )


def downgrade() -> None:
    op.drop_table("videos")
    op.drop_table("segments")
    op.drop_table("jobs")
    op.drop_table("users")
