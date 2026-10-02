"""Characters: allow a draft with neither a LoRA nor a reference (wanly-console#592)

Revision ID: 110
Revises: 109
Create Date: 2026-10-02

Migration 107 required every character to have a LoRA or a sheet/face reference
(ck_ltx_characters_lora_or_ref). But Build sheet (console#582/#585) works on an EXISTING
character, so a brand-new, LoRA-less character could never get its first sheet: it could not
be saved without one, and the sheet could not be built without it being saved.

So the CHECK goes. A row with neither is a DRAFT; the API refuses to render one (submit and
claim, "<name> has no LoRA or sheet yet") instead of the database refusing to store it.
ck_ltx_characters_identity_mode stays: a mode must still name a reference that is set.

NO DATA IS CHANGED: dropping a CHECK only allows more rows. The downgrade puts the CHECK back,
which first needs every draft to pass it, so it gives a draft the legacy "none" LoRA -- what a
registration ahead of training (#352) stored, and what migration 107's CHECK accepted.
"""
from alembic import op

revision = "110"
down_revision = "109"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_constraint("ck_ltx_characters_lora_or_ref", "ltx_characters", type_="check")


def downgrade() -> None:
    op.execute("UPDATE ltx_characters SET char_lora = 'none' "
               "WHERE char_lora IS NULL AND sheet_uri IS NULL AND face_ref_uri IS NULL")
    op.create_check_constraint(
        "ck_ltx_characters_lora_or_ref", "ltx_characters",
        "char_lora IS NOT NULL OR sheet_uri IS NOT NULL OR face_ref_uri IS NOT NULL")
