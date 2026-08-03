"""Add identity trajectory endpoints to segments

Revision ID: 054
Revises: 053
Create Date: 2026-08-03

The mean cosine blurs the shape of the drift. A segment going 0.95 -> 0.65 averages about
the same as one sitting flat at 0.80, and only the first has lost the character.

These record the FIRST and LAST frame's cosine against the job's ground truth (the job's
start image, which is the same reference for every segment). Loss across a segment is
start - end, and because a continuation begins where the previous segment ended, the
endpoints chain across the whole job:

    seg0   0.98 -> 0.85    loss 0.13
    seg1   0.85 -> 0.60    loss 0.25
    job    0.98 -> 0.60

That chain is what makes "which segment lost her, and how much" answerable.
"""
import sqlalchemy as sa
from alembic import op

revision = "054"
down_revision = "053"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("segments", sa.Column("identity_start_cos_ref", sa.Float(), nullable=True))
    op.add_column("segments", sa.Column("identity_end_cos_ref", sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("segments", "identity_end_cos_ref")
    op.drop_column("segments", "identity_start_cos_ref")
