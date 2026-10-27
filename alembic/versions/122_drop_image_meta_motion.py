"""Drop image_meta's motion caption columns (wanly-api#481)

The motion caption half -- a second paragraph per image, made by Qwen3-VL and filled into a
pose's <MOTION> -- was removed. Its three columns go with it. The scene half
(scene_description / scene_instruction / scene_described_at) is untouched.

Downgrade re-adds the columns nullable and EMPTY: the paragraphs are gone for good once this
runs, which is fine -- nothing reads them, and a prompt that still carries <MOTION> is
stripped at submit and at the claim rather than filled.

Revision ID: 122
Revises: 121
"""
import sqlalchemy as sa
from alembic import op

revision = "122"
down_revision = "121"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # op.drop_column, not raw SQL: tests/test_dropped_columns_are_gone.py reads these calls
    # and fails if app/ still names a dropped column anywhere.
    op.drop_column("image_meta", "motion_description")
    op.drop_column("image_meta", "motion_instruction")
    op.drop_column("image_meta", "motion_described_at")


def downgrade() -> None:
    op.add_column("image_meta", sa.Column("motion_description", sa.Text(), nullable=True))
    op.add_column("image_meta", sa.Column("motion_instruction", sa.Text(), nullable=True))
    op.add_column("image_meta",
                  sa.Column("motion_described_at", sa.DateTime(timezone=True), nullable=True))
