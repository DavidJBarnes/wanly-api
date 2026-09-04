"""Per-pose base model (checkpoint)

Revision ID: 081
Revises: 080
Create Date: 2026-09-03

NULLABLE, and null means "use the stack's value" — the same shape as frames,
negative_prompt, img_compression and content_lora. Every existing pose keeps rendering on
sulphur_dev_bf16, so this migration changes no output.

WHY PER POSE
    `checkpoint` was a global in LTX_STACK, so every render used one base model and there
    was no way to compare two. Four sit on the 3090 — sulphur_dev_bf16, 10Eros_v1.5_bf16,
    ltx-2.3-22b-dev, ltx-2.3-22b-distilled-1.1 — and only one was reachable.

THE RISK, STATED
    Character LoRAs were trained against sulphur. Against a different base, a LoRA whose
    keys do not line up fuses NOTHING and says nothing about it — the engine's own
    lora_coverage() docstring records that there is no error, no warning and no log line,
    and the run looks completely normal. The render then comes back as the base model with
    none of the character in it.

    That is accepted deliberately here: comparing base models is the point. What this change
    also does is make the fusion count TRUE for the checkpoint actually in use, so a
    comparison can be read rather than guessed at.
"""
import sqlalchemy as sa
from alembic import op

revision = "081"
down_revision = "080"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("ltx_recipes", sa.Column("checkpoint", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("ltx_recipes", "checkpoint")
