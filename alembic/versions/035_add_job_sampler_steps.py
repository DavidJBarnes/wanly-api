"""Add per-job sampler steps: jobs.steps_total + jobs.high_noise_steps

Revision ID: 035
Revises: 030
Create Date: 2026-07-07

Per-job override for the KSampler schedule so the UI can drive de-distilled experiments
(lightx2v strength 0 drops the Lightning LoRA in the daemon builder; more steps + higher
CFG then produce real guidance). Both nullable -> daemon defaults (4 / 2) when unset.

Numbered 035 (not 031) to leave a gap over the orphaned 031-034 revision IDs from the
identity-pursuit rollback, so it can never collide if a reverted branch ever resurfaces.
"""

import sqlalchemy as sa
from alembic import op

revision = "035"
down_revision = "030"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("jobs", sa.Column("steps_total", sa.Integer(), nullable=True))
    op.add_column("jobs", sa.Column("high_noise_steps", sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column("jobs", "high_noise_steps")
    op.drop_column("jobs", "steps_total")
