"""A character carries the gender its LoRA trained on

Revision ID: 096
Revises: 095
Create Date: 2026-09-09

Every LoRA trains on the caption "<trigger>, <gender>" (093/#293), but a pose's <TRIGGER>
was filled with the bare trigger, so the word the identity was bound to never reached a
render prompt (wanly-console#487). The column is what lets the fill say "p@yton, woman".

Backfilled from each character's latest completed training run, which is the only record
of what actually trained. Characters that predate the trainer stay NULL and render exactly
as they did; the character dialog can set them.
"""
import sqlalchemy as sa
from alembic import op

revision = "096"
down_revision = "095"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("ltx_characters", sa.Column("gender", sa.String(16), nullable=True))
    op.execute("""
        UPDATE ltx_characters c
        SET gender = t.gender
        FROM (
            SELECT DISTINCT ON (character) character, config->>'gender' AS gender
            FROM training_jobs
            WHERE status = 'completed' AND config->>'gender' IS NOT NULL
            ORDER BY character, created_at DESC
        ) t
        WHERE c.name = t.character AND c.gender IS NULL
    """)


def downgrade() -> None:
    op.drop_column("ltx_characters", "gender")
