"""Add ltx_characters.trained_from: which datasets trained this LoRA

Revision ID: 099
Revises: 098
Create Date: 2026-09-13

A character row names its LoRA but not where the LoRA came from. The training job has
always known (dataset_images + identities), but the dataset NAMES were discarded at
creation and the row was never stamped. `trained_from` is the stamp:

    [{dataset_id, name, count}, ...]   # group order: group 0 first, then identities

Snapshotted at PUBLISH time from what creation recorded: a dataset renamed later does not
rewrite what trained, and a deleted dataset keeps its name (the id is then dangling and
the console renders the entry as plain text).

NULL for every character published before this migration -- they are backfilled by a
one-off pass keyed on their training jobs.
"""
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "099"
down_revision = "098"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "ltx_characters",
        sa.Column("trained_from", postgresql.JSONB(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("ltx_characters", "trained_from")
