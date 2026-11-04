"""image_meta.describe_queued_at / describe_error (console#727)

The scene describer's queue, on the image's own row so it survives a deploy.

Revision ID: 126
Revises: 125
"""
import sqlalchemy as sa
from alembic import op

revision = "126"
down_revision = "125"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("image_meta",
                  sa.Column("describe_queued_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("image_meta", sa.Column("describe_error", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("image_meta", "describe_error")
    op.drop_column("image_meta", "describe_queued_at")
