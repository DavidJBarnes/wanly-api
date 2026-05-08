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

    # Backfill tags from the first completed video for existing jobs
    op.execute(
        """
        UPDATE jobs
        SET tags = sub.tags
        FROM (
            SELECT DISTINCT ON (videos.job_id) videos.job_id, videos.tags
            FROM videos
            WHERE videos.status = 'completed' AND videos.tags IS NOT NULL
            ORDER BY videos.job_id, videos.completed_at
        ) sub
        WHERE jobs.id = sub.job_id
        """
    )


def downgrade() -> None:
    op.drop_index("ix_jobs_tags")
    op.drop_column("jobs", "tags")
