"""Add Final Cut columns: jobs.kind + jobs.source_job_id, segments.animate_mode/preset

Revision ID: 033
Revises: 032
Create Date: 2026-07-04

Final Cut re-renders a job's finalized video through Wan Animate as a linked
child job. A `final_cut` job carries `kind="final_cut"` and `source_job_id`
pointing at its source (lineage, both directions). Its single segment uses
`reprocess_type="animate"` and stores the Animate `animate_mode` (move/mix) and
`animate_preset` (fast/highres). All columns are nullable / server-defaulted so
existing rows backfill cleanly (kind -> 'generate').
"""

import sqlalchemy as sa
from alembic import op

revision = "033"
down_revision = "032"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "jobs",
        sa.Column("kind", sa.String(length=20), nullable=False, server_default="generate"),
    )
    op.add_column("jobs", sa.Column("source_job_id", sa.dialects.postgresql.UUID(as_uuid=True), nullable=True))
    op.create_index("ix_jobs_source_job_id", "jobs", ["source_job_id"])
    op.create_foreign_key(
        "fk_jobs_source_job_id", "jobs", "jobs",
        ["source_job_id"], ["id"], ondelete="SET NULL",
    )
    op.add_column("segments", sa.Column("animate_mode", sa.String(length=20), nullable=True))
    op.add_column("segments", sa.Column("animate_preset", sa.String(length=20), nullable=True))


def downgrade() -> None:
    op.drop_column("segments", "animate_preset")
    op.drop_column("segments", "animate_mode")
    op.drop_constraint("fk_jobs_source_job_id", "jobs", type_="foreignkey")
    op.drop_index("ix_jobs_source_job_id", table_name="jobs")
    op.drop_column("jobs", "source_job_id")
    op.drop_column("jobs", "kind")
