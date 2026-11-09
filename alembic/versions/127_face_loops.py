"""face_loops (wanly-api#495, wanly-console#742)

Per-character AR face loops: the render job that made each one, its preparation status, and
where the prepared loop + mesh data live.

Revision ID: 127
Revises: 126
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "127"
down_revision = "126"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "face_loops",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("character_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("ltx_characters.id", ondelete="CASCADE"), nullable=False),
        sa.Column("label", sa.String(120), nullable=False),
        sa.Column("preset", sa.String(32), nullable=False),
        sa.Column("prompt", sa.Text(), nullable=False),
        sa.Column("start_image", sa.Text(), nullable=False),
        sa.Column("job_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("jobs.id", ondelete="SET NULL"), nullable=True),
        sa.Column("status", sa.String(16), nullable=False, server_default="rendering"),
        sa.Column("asset_prefix", sa.Text(), nullable=True),
        sa.Column("window_start", sa.Float(), nullable=True),
        sa.Column("window_end", sa.Float(), nullable=True),
        sa.Column("stats", postgresql.JSONB(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_face_loops_character_id", "face_loops", ["character_id"])
    op.create_index("ix_face_loops_job_id", "face_loops", ["job_id"])


def downgrade() -> None:
    op.drop_index("ix_face_loops_job_id", table_name="face_loops")
    op.drop_index("ix_face_loops_character_id", table_name="face_loops")
    op.drop_table("face_loops")
