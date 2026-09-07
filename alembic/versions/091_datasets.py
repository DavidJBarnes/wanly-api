"""Named, taggable training datasets

Revision ID: 091
Revises: 090
Create Date: 2026-09-07

The grouping that did not exist. Before this the only ways to say "these images belong together"
were an S3 folder prefix and a per-user favourites list, so a training set could not be named,
tagged, or re-opened -- every run meant re-selecting the images by hand, and a v2 meant doing it
again from memory.

Images are an explicit ORDERED list rather than a folder listing. A dataset usually corresponds
to a folder and uploads go into one, but the list is what the dataset IS: it survives an image
being moved, and it fixes the order the trainer stages them in, which is what the captions pair
against.

`tags` is a comma-separated string rather than an array, matching ImageMeta.tags. A second
convention for the same idea would mean the console parsing tags two ways.
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "091"
down_revision = "090"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "datasets",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=True),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("tags", sa.String(length=500), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("images", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("prefix", sa.String(length=200), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_datasets_name", "datasets", ["name"], unique=True)


def downgrade() -> None:
    op.drop_index("ix_datasets_name", table_name="datasets")
    op.drop_table("datasets")
