"""Add archived flag to video settings presets

Revision ID: 056
Revises: 055
Create Date: 2026-08-05

Presets accumulate. Two days of identity experiments created ~36 throwaway presets against 8
real ones, burying the recipes that matter in a list of one-off test arms.

Deleting them is not an option: jobs reference a preset by id, so the historical record of what
config produced which result would be lost — and that record is the entire value of the
experiments. Same reasoning as archiving jobs rather than deleting them.

Archived presets stay fully readable by id, so every past job still resolves its config. They
are only hidden from the list used to PICK a preset, which is where the clutter hurts.
"""
import sqlalchemy as sa
from alembic import op

revision = "056"
down_revision = "055"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "video_settings_presets",
        sa.Column("archived", sa.Boolean(), nullable=False, server_default=sa.false()),
    )


def downgrade() -> None:
    op.drop_column("video_settings_presets", "archived")
