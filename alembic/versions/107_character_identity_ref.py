"""Characters: a LoRA, a sheet, or both (wanly-api#379, epic wanly-console#581)

Revision ID: 107
Revises: 106
Create Date: 2026-10-01

Phase 0 (wanly-gpu-docker#155) showed a 1536x1024 character sheet, conditioned into wanly's
own recipe graph, holding identity better than the character LoRA alone. So a character is
no longer "a LoRA and a trigger": it can be a LoRA, an identity reference, or both.

  ltx_characters.char_lora      NOT NULL -> nullable. NULL = this character has no LoRA (a
                                sheet-only character). The legacy "none" string still means
                                the same thing and is left exactly where it is.
  ltx_characters.trigger        NOT NULL -> nullable. A sheet-only character has no caption
                                trigger; <TRIGGER> fills from `description` instead.
  ltx_characters.sheet_uri      the 1536x1024 character sheet (an Image Repo s3:// URI)
  ltx_characters.face_ref_uri   an optional face close-up
  ltx_characters.identity_mode  'sheet' | 'face' | NULL: which reference renders
  ltx_characters.description    short words for <TRIGGER> when there is no trigger
  jobs.use_identity_ref         per-job toggle; NULL = the default, which is "on when the
                                character has a reference"

Two CHECKs: a character has a LoRA or a reference (char_lora IS NOT NULL counts the legacy
"none" rows, so every existing row passes untouched), and identity_mode names a reference
that is actually set.

NO DATA IS CHANGED. Every existing row keeps its LoRA, trigger and strengths; the new columns
are NULL. The downgrade has to put values back into the two NOT NULL columns, so it fills a
NULL char_lora with "none" and a NULL trigger with the name -- exactly what create_character
would have stored for them before this migration.
"""
import sqlalchemy as sa

from alembic import op

revision = "107"
down_revision = "106"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column("ltx_characters", "char_lora", existing_type=sa.Text(), nullable=True)
    op.alter_column("ltx_characters", "trigger", existing_type=sa.String(255), nullable=True)
    op.add_column("ltx_characters", sa.Column("sheet_uri", sa.Text(), nullable=True))
    op.add_column("ltx_characters", sa.Column("face_ref_uri", sa.Text(), nullable=True))
    op.add_column("ltx_characters", sa.Column("identity_mode", sa.String(8), nullable=True))
    op.add_column("ltx_characters", sa.Column("description", sa.String(255), nullable=True))
    op.create_check_constraint(
        "ck_ltx_characters_lora_or_ref", "ltx_characters",
        "char_lora IS NOT NULL OR sheet_uri IS NOT NULL OR face_ref_uri IS NOT NULL")
    op.create_check_constraint(
        "ck_ltx_characters_identity_mode", "ltx_characters",
        "identity_mode IS NULL"
        " OR (identity_mode = 'sheet' AND sheet_uri IS NOT NULL)"
        " OR (identity_mode = 'face' AND face_ref_uri IS NOT NULL)")
    op.add_column("jobs", sa.Column("use_identity_ref", sa.Boolean(), nullable=True))


def downgrade() -> None:
    op.drop_column("jobs", "use_identity_ref")
    op.drop_constraint("ck_ltx_characters_identity_mode", "ltx_characters", type_="check")
    op.drop_constraint("ck_ltx_characters_lora_or_ref", "ltx_characters", type_="check")
    op.drop_column("ltx_characters", "description")
    op.drop_column("ltx_characters", "identity_mode")
    op.drop_column("ltx_characters", "face_ref_uri")
    op.drop_column("ltx_characters", "sheet_uri")
    op.execute("UPDATE ltx_characters SET trigger = name WHERE trigger IS NULL")
    op.execute("UPDATE ltx_characters SET char_lora = 'none' WHERE char_lora IS NULL")
    op.alter_column("ltx_characters", "trigger", existing_type=sa.String(255), nullable=False)
    op.alter_column("ltx_characters", "char_lora", existing_type=sa.Text(), nullable=False)
