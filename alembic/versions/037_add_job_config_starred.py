"""Add jobs.config_starred

Revision ID: 037
Revises: 036
Create Date: 2026-07-09

Per-job bookmark flag: the user marks a job whose runtime config produced a good result
as a "successful config" so it can be found later (Job Queue starred filter). Non-null,
defaults false for existing rows.
"""
import sqlalchemy as sa
from alembic import op

revision = "037"
down_revision = "036"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "jobs",
        sa.Column("config_starred", sa.Boolean(), nullable=False, server_default=sa.false()),
    )


def downgrade() -> None:
    op.drop_column("jobs", "config_starred")
