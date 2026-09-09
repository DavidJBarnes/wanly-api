"""Notes on a training run

Revision ID: 095
Revises: 094
Create Date: 2026-09-09

A free-form operator warning on TrainingJob (wanly-console#484), exactly as Dataset.notes
carries it: what was learned by eye -- why a checkpoint was picked, what was rejected.
Written only through the console's own route; the trainer's report channel cannot reach it.
"""
import sqlalchemy as sa
from alembic import op

revision = "095"
down_revision = "094"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("training_jobs", sa.Column("notes", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("training_jobs", "notes")
