"""Add TrainingJob.second_identity for joint two-identity runs

Revision ID: 097
Revises: 096
Create Date: 2026-09-11

A joint run trains ONE LoRA on BOTH characters' datasets simultaneously (#102): the
group-0 columns (character, trigger, dataset_images) stay what they always were, and the
second identity rides a nullable JSONB beside them. NULL for every run before this --
single-identity is the unchanged default, nothing is rewritten.

The column is JSONB, not columns: the second group is one read in one place (the claim
builder), and the images list already has a JSONB precedent in dataset_images. Splitting
it into five nullable columns would be five chances for half a second identity -- a run
with a trigger but no images builds a half-configured dataset silently.
"""
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "097"
down_revision = "096"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "training_jobs",
        sa.Column("second_identity", postgresql.JSONB(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("training_jobs", "second_identity")
