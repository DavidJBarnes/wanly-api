"""Add priority column to jobs table

Revision ID: 007
Revises: 006
Create Date: 2026-02-19
"""

from alembic import op
import sqlalchemy as sa

revision = "007"
down_revision = "006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("jobs", sa.Column("priority", sa.Integer(), nullable=False, server_default="0"))
    op.create_index("ix_jobs_priority", "jobs", ["priority"])


def downgrade() -> None:
    op.drop_index("ix_jobs_priority", table_name="jobs")
    op.drop_column("jobs", "priority")
