"""Character sheets: provenance of each generated sheet (wanly-api#380, epic wanly-console#582)

Revision ID: 108
Revises: 107
Create Date: 2026-10-01

The console now builds a character's 1536x1024 sheet from a real face photo (Qwen-Image-Edit-2511
turnaround + the real face panel). Approving a candidate saves the sheet to the Image Repo and
sets ltx_characters.sheet_uri / identity_mode -- columns migration 107 already added. What 107
has no place for is WHERE the sheet came from: the face photo, the outfit/hair/body words, the
prompt, the seed and the model. That is this table, one row per saved sheet.

A NEW TABLE ONLY. No existing row of any table is read or changed. character_id is SET NULL on
delete (the name is kept): the sheet image outlives the character in the repo, and so does its
record.
"""
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "108"
down_revision = "107"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "character_sheets",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("character_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("ltx_characters.id", ondelete="SET NULL"), nullable=True),
        sa.Column("character_name", sa.String(64), nullable=False),
        sa.Column("sheet_uri", sa.Text(), nullable=False),
        sa.Column("candidate_uri", sa.Text(), nullable=True),
        sa.Column("face_uri", sa.Text(), nullable=False),
        sa.Column("outfit", sa.Text(), nullable=False),
        sa.Column("hair", sa.Text(), nullable=True),
        sa.Column("body", sa.Text(), nullable=True),
        sa.Column("gender", sa.String(16), nullable=True),
        sa.Column("prompt", sa.Text(), nullable=False),
        sa.Column("seed", sa.BigInteger(), nullable=False),
        sa.Column("model", sa.Text(), nullable=True),
        sa.Column("settings", sa.Text(), nullable=True),
        sa.Column("files", postgresql.JSONB(), nullable=True),
        sa.Column("face_panel", sa.String(16), nullable=True),
        sa.Column("identity", postgresql.JSONB(), nullable=True),
        sa.Column("job_id", sa.String(64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_character_sheets_character_id", "character_sheets", ["character_id"])


def downgrade() -> None:
    op.drop_index("ix_character_sheets_character_id", table_name="character_sheets")
    op.drop_table("character_sheets")
