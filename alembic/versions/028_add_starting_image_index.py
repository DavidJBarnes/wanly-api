"""Add index on jobs.starting_image

Revision ID: 028
Revises: 027
Create Date: 2026-05-08
"""

from alembic import op

revision = "028"
down_revision = "027"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index("ix_jobs_starting_image", "jobs", ["starting_image"])


def downgrade() -> None:
    op.drop_index("ix_jobs_starting_image")
