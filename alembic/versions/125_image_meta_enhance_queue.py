"""image_meta.enhance_queued_at / enhance_error (console#719)

The prompt enhancer's queue, on the image's own row so it survives a deploy.

Revision ID: 125
Revises: 124
"""
import sqlalchemy as sa
from alembic import op

revision = "125"
down_revision = "124"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("image_meta",
                  sa.Column("enhance_queued_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("image_meta", sa.Column("enhance_error", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("image_meta", "enhance_error")
    op.drop_column("image_meta", "enhance_queued_at")
