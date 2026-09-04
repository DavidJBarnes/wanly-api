"""Several content LoRAs per pose, each with its own per-stage strengths

Revision ID: 083
Revises: 082
Create Date: 2026-09-04

console#410. Replaces the single content_lora + content_s1 + content_s2 with a JSONB list of
{name, s1, s2}, because three scalar columns do not extend to N.

THE MIGRATION IS THE POINT
    Every existing pose must render exactly as it does now. The upgrade folds each pose's
    current three values into a ONE-ELEMENT list, and the downgrade takes the first element
    back out. A pose with no content LoRA becomes an empty list, which resolves to "none"
    exactly as NULL did.

ORDER IS SIGNIFICANT
    A JSONB array preserves order, and that order is the order the LoRAs are applied in the
    chain. It is not a set. Two poses with the same LoRAs in a different order are different
    configurations and will render differently.
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "083"
down_revision = "082"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("ltx_recipes",
                  sa.Column("content_loras", postgresql.JSONB(), nullable=True))
    # Fold the existing single LoRA into a one-element list. "none" and empty are not a
    # LoRA and become an empty list rather than a list containing nothing useful — the
    # engine reads both as "render without one", and a [{"name": "none"}] would be a
    # filename lookup for a file that does not exist.
    op.execute("""
        UPDATE ltx_recipes
           SET content_loras = jsonb_build_array(
                 jsonb_build_object(
                   'name', content_lora,
                   's1', COALESCE(content_s1, 0.6),
                   's2', COALESCE(content_s2, 0.6)))
         WHERE content_lora IS NOT NULL
           AND btrim(content_lora) <> ''
           AND lower(btrim(content_lora)) <> 'none'
    """)
    op.execute("UPDATE ltx_recipes SET content_loras = '[]'::jsonb WHERE content_loras IS NULL")
    op.drop_column("ltx_recipes", "content_lora")
    op.drop_column("ltx_recipes", "content_s1")
    op.drop_column("ltx_recipes", "content_s2")


def downgrade() -> None:
    op.add_column("ltx_recipes", sa.Column("content_lora", sa.Text(), nullable=True))
    op.add_column("ltx_recipes", sa.Column("content_s1", sa.Float(), nullable=True))
    op.add_column("ltx_recipes", sa.Column("content_s2", sa.Float(), nullable=True))
    # Only the FIRST survives — the old shape cannot hold more. Lossy by nature, and that is
    # the honest behaviour: silently keeping one of four is better than failing the
    # downgrade, but it is why this direction should be rare.
    op.execute("""
        UPDATE ltx_recipes
           SET content_lora = content_loras->0->>'name',
               content_s1   = (content_loras->0->>'s1')::float,
               content_s2   = (content_loras->0->>'s2')::float
         WHERE jsonb_array_length(COALESCE(content_loras, '[]'::jsonb)) > 0
    """)
    op.drop_column("ltx_recipes", "content_loras")
