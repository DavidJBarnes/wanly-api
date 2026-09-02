"""Per-pose content LoRA and its two stage strengths

Revision ID: 079
Revises: 078
Create Date: 2026-09-02

All three NULLABLE, and null means "use the stack's value" — the same shape as
`negative_prompt`, `frames` and `img_compression`, which resolve as
`r.frames or LTX_STACK["frames"]`. Every existing pose keeps rendering exactly as it does
now, so this migration changes no output.

WHY THIS BELONGS ON A POSE AND THE CHARACTER LORA DOES NOT
----------------------------------------------------------
The graph has always chained two LoRAs per stage:

    content LoRA -> character LoRA -> branch

They answer different questions. The character LoRA is WHO — identity, which is a property
of the character and lives on ltx_characters. The content LoRA is WHAT IS HAPPENING —
motion and act, which is a property of the POSE and was until now a single global
(LTX_STACK["content_lora"], pinned to "none"). A global cannot say "sfbehind for the
from-behind poses and nothing for the rest", which is the whole point of having them.

Note that "none" stays the default. Dropping DR34ML4Y is what removed the motion horror,
and this must not quietly reintroduce a content LoRA anywhere it is not asked for: a NULL
here means the stack value, and the stack value is still "none".

WHY TWO STRENGTHS AND NOT ONE
-----------------------------
resolve() hardcoded the content strength to 0.6 for BOTH stages while character LoRAs got
separate s1/s2. Stage 1 generates at half size from noise; stage 2 refines the 2x-upscaled
latent and is where detail resolves. Collapsing them is a different configuration, not a
simplification — the same reasoning already recorded on LtxCharacter.strength_stage_1/2.
"""
import sqlalchemy as sa
from alembic import op

revision = "079"
down_revision = "078"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("ltx_recipes", sa.Column("content_lora", sa.Text(), nullable=True))
    op.add_column("ltx_recipes", sa.Column("content_s1", sa.Float(), nullable=True))
    op.add_column("ltx_recipes", sa.Column("content_s2", sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("ltx_recipes", "content_s2")
    op.drop_column("ltx_recipes", "content_s1")
    op.drop_column("ltx_recipes", "content_lora")
