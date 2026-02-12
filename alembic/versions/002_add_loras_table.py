"""Add loras table

Revision ID: 002
Revises: 001
Create Date: 2026-02-12
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

revision = "002"
down_revision = "001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "loras",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("trigger_words", sa.Text, nullable=True),
        sa.Column("default_prompt", sa.Text, nullable=True),
        sa.Column("source_url", sa.Text, nullable=True),
        sa.Column("preview_image", sa.Text, nullable=True),
        sa.Column("high_file", sa.String(255), nullable=True),
        sa.Column("high_s3_uri", sa.Text, nullable=True),
        sa.Column("low_file", sa.String(255), nullable=True),
        sa.Column("low_s3_uri", sa.Text, nullable=True),
        sa.Column("default_high_weight", sa.Float, nullable=False, server_default="1.0"),
        sa.Column("default_low_weight", sa.Float, nullable=False, server_default="1.0"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )


def downgrade() -> None:
    op.drop_table("loras")
