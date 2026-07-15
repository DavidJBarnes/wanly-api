"""Add smashcut carrier columns to segments

Revision ID: 047
Revises: 046
Create Date: 2026-07-14

Foundry smashcut: a carrier segment (reprocess_type="smashcut_concat") holds the ordered list
of source clip paths and the transition style; the daemon concatenates them into one montage.
"""
import sqlalchemy as sa
from alembic import op

revision = "047"
down_revision = "046"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("segments", sa.Column("smashcut_clip_paths", sa.JSON(), nullable=True))
    op.add_column("segments", sa.Column("smashcut_transition", sa.String(length=20), nullable=True))


def downgrade() -> None:
    op.drop_column("segments", "smashcut_transition")
    op.drop_column("segments", "smashcut_clip_paths")
