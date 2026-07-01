"""Add jobs.mode (generation mode preset), drop per-job lightx2v/cfg knobs

Revision ID: 032
Revises: 031
Create Date: 2026-07-01

The New Job modal's raw LightX2V/CFG knobs are replaced by a single `mode`
selector (Wan22 Base Character Identity / Wan22 Base Identity+Expression /
DaSiWa Fast). The daemon resolves each mode to a full model+sampler preset,
so the per-job lightx2v/cfg columns are no longer used. `mode` is added with a
server_default so existing rows backfill to 'identity' (the safe fast preset).
"""

import sqlalchemy as sa
from alembic import op

revision = "032"
down_revision = "031"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "jobs",
        sa.Column("mode", sa.String(length=20), nullable=False, server_default="identity"),
    )
    op.drop_column("jobs", "lightx2v_strength_high")
    op.drop_column("jobs", "lightx2v_strength_low")
    op.drop_column("jobs", "cfg_high")
    op.drop_column("jobs", "cfg_low")
    # Stale global defaults are no longer read by the API; remove them if present.
    op.execute(
        "DELETE FROM app_settings WHERE key IN "
        "('cfg_high','cfg_low','lightx2v_strength_high','lightx2v_strength_low')"
    )


def downgrade() -> None:
    op.add_column("jobs", sa.Column("lightx2v_strength_high", sa.Float(), nullable=True))
    op.add_column("jobs", sa.Column("lightx2v_strength_low", sa.Float(), nullable=True))
    op.add_column("jobs", sa.Column("cfg_high", sa.Float(), nullable=True))
    op.add_column("jobs", sa.Column("cfg_low", sa.Float(), nullable=True))
    op.drop_column("jobs", "mode")
