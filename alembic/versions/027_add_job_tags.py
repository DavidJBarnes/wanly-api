"""Add tags column to jobs

Revision ID: 027
Revises: 026
Create Date: 2026-05-08
"""

from alembic import op
import sqlalchemy as sa

revision = "027"
down_revision = "026"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "jobs",
        sa.Column("tags", sa.Text(), nullable=True),
    )
    op.create_index("ix_jobs_tags", "jobs", ["tags"])


def downgrade() -> None:
    op.drop_index("ix_jobs_tags")
    op.drop_column("jobs", "tags")
