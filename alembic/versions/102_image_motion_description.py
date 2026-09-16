"""Image motion descriptions (wanly-api#326)

Revision ID: 102
Revises: 101
Create Date: 2026-09-16

The tagging call now produces a second caption: the frame read as the first frame of a
10-second clip, for the motion half of an image-to-video prompt. Produced in the same
session as scene_description and grounded on it.

Nullable with no backfill: the whole existing library was described by JoyCaption, which
cannot do this half, and re-describing every image would be GPU work nobody asked for.
Null means "described before #326", which the console renders as an absent section, not a
failure.
"""
import sqlalchemy as sa
from alembic import op

revision = "102"
down_revision = "101"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("image_meta", sa.Column("motion_description", sa.Text(), nullable=True))
    op.add_column("image_meta", sa.Column("motion_instruction", sa.Text(), nullable=True))
    op.add_column("image_meta",
                  sa.Column("motion_described_at", sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    op.drop_column("image_meta", "motion_described_at")
    op.drop_column("image_meta", "motion_instruction")
    op.drop_column("image_meta", "motion_description")
