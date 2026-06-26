"""Add sampler params (steps_total, high_noise_steps, shift_high, shift_low) to segments

Revision ID: 031
Revises: 030
Create Date: 2026-06-26
"""

import sqlalchemy as sa
from alembic import op

revision = "031"
down_revision = "030"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("segments", sa.Column("steps_total", sa.Integer(), nullable=True))
    op.add_column("segments", sa.Column("high_noise_steps", sa.Integer(), nullable=True))
    op.add_column("segments", sa.Column("shift_high", sa.Float(), nullable=True))
    op.add_column("segments", sa.Column("shift_low", sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("segments", "shift_low")
    op.drop_column("segments", "shift_high")
    op.drop_column("segments", "high_noise_steps")
    op.drop_column("segments", "steps_total")
