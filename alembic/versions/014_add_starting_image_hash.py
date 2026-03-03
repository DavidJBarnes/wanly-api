"""Add starting_image_hash to jobs

Revision ID: 014
Revises: 013
Create Date: 2026-03-03
"""

from alembic import op
import sqlalchemy as sa

revision = "014"
down_revision = "013"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("jobs", sa.Column("starting_image_hash", sa.String(64), nullable=True))
    op.create_index("ix_jobs_starting_image_hash", "jobs", ["starting_image_hash"])


def downgrade() -> None:
    op.drop_index("ix_jobs_starting_image_hash", table_name="jobs")
    op.drop_column("jobs", "starting_image_hash")
