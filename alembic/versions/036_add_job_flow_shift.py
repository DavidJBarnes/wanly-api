"""Add jobs.flow_shift

Revision ID: 036
Revises: 035
Create Date: 2026-07-07

Per-job flow-matching schedule shift (ModelSamplingSD3). Higher = more high-noise steps =
more motion; the lever for the de-distilled 'frozen' look. Nullable -> daemon default (5.0).
"""
import sqlalchemy as sa
from alembic import op

revision = "036"
down_revision = "035"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("jobs", sa.Column("flow_shift", sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("jobs", "flow_shift")
