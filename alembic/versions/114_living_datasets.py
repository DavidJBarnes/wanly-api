"""Living datasets: datasets.archived_at, training_run_datasets (wanly-api#419)

A set no longer locks once it trains (#420), so the RUN becomes the record of what trained.
Every run already snapshots its images and captions per group in its own columns -- shaped
for the trainer, group 0 flat and the rest in `identities`. This table restates them one row
per group, keyed by the dataset each came from, so "what did v3 train on" and "which runs used
this image" are a query instead of a walk through JSON:

  training_run_datasets   one row per training group: the run, the dataset (NULL when the
                          group had none, or the set was deleted), and the group exactly as it
                          trained -- images, captions, repeats, windows

  datasets.archived_at    a version set folded into its subject's living set by the backfill
                          (#424): hidden from lists and pickers, read-only, never deleted

Additive. Rows for runs created before this are written by the backfill, not here: deciding
which dataset an early run used needs judgement the migration should not make on deploy.

Revision ID: 114
Revises: 113
Create Date: 2026-10-08
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "114"
down_revision = "113"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("datasets", sa.Column("archived_at", sa.DateTime(timezone=True), nullable=True))
    op.create_table(
        "training_run_datasets",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("training_job_id", UUID(as_uuid=True),
                  sa.ForeignKey("training_jobs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("group_index", sa.Integer(), nullable=False),
        sa.Column("dataset_id", UUID(as_uuid=True),
                  sa.ForeignKey("datasets.id", ondelete="SET NULL"), nullable=True),
        sa.Column("dataset_name", sa.String(100), nullable=True),
        sa.Column("kind", sa.String(20), nullable=False),
        sa.Column("character", sa.String(64), nullable=True),
        sa.Column("images", JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("captions", JSONB(), nullable=True),
        sa.Column("num_repeats", sa.Integer(), nullable=True),
        sa.Column("windows", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("source", sa.String(20), nullable=False, server_default="created"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("now()")),
        sa.UniqueConstraint("training_job_id", "group_index",
                            name="uq_training_run_datasets_group"),
    )
    op.create_index("ix_training_run_datasets_job", "training_run_datasets", ["training_job_id"])
    op.create_index("ix_training_run_datasets_dataset", "training_run_datasets", ["dataset_id"])


def downgrade() -> None:
    op.drop_index("ix_training_run_datasets_dataset", "training_run_datasets")
    op.drop_index("ix_training_run_datasets_job", "training_run_datasets")
    op.drop_table("training_run_datasets")
    op.drop_column("datasets", "archived_at")
