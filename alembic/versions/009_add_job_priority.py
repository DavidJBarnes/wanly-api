"""Add priority column to jobs table

Revision ID: 009
Revises: 008
Create Date: 2026-02-23
"""

from alembic import op
import sqlalchemy as sa

revision = "009"
down_revision = "008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("jobs", sa.Column("priority", sa.Integer(), nullable=False, server_default="0"))
    op.create_index("ix_jobs_priority", "jobs", ["priority"])

    # Backfill existing rows: assign priority by created_at order (oldest = 0 = highest priority)
    op.execute("""
        UPDATE jobs SET priority = sub.rn FROM (
            SELECT id, ROW_NUMBER() OVER (ORDER BY created_at ASC) - 1 AS rn
            FROM jobs
        ) sub WHERE jobs.id = sub.id
    """)


def downgrade() -> None:
    op.drop_index("ix_jobs_priority", table_name="jobs")
    op.drop_column("jobs", "priority")
