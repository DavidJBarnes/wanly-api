"""image_meta.enhanced_prompt / enhanced_prompt_at (wanly-api#480)

The Sulphur-2 prompt enhancer's video prompt for an image, kept like the scene description.

Revision ID: 123
Revises: 122
"""
import sqlalchemy as sa
from alembic import op

revision = "123"
down_revision = "122"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("image_meta", sa.Column("enhanced_prompt", sa.Text(), nullable=True))
    op.add_column("image_meta",
                  sa.Column("enhanced_prompt_at", sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    op.drop_column("image_meta", "enhanced_prompt_at")
    op.drop_column("image_meta", "enhanced_prompt")
