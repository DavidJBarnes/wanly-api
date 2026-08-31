"""Drop Job.config_starred

Revision ID: 076
Revises: 075
Create Date: 2026-08-31

The last of the Configurations feature. It starred a job's WAN sampler settings — lightx2v,
cfg high/low, steps, flow shift — so a combination that worked could be found again. Those
settings were dropped in 075, so the flag pointed at nothing.

Under LTX there is nothing to star. The stack is one global configuration and a recipe is
(character, prompt), so a validated pose is already a first-class row in `ltx_recipes` with a
`validated` flag — which is what starring was approximating.

Nullable-free and non-destructive in the sense that matters: the column held a boolean opinion
about settings that no longer exist.
"""
import sqlalchemy as sa
from alembic import op

revision = "076"
down_revision = "075"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_column("jobs", "config_starred")


def downgrade() -> None:
    op.add_column(
        "jobs",
        sa.Column("config_starred", sa.Boolean(), nullable=False, server_default="false"),
    )
