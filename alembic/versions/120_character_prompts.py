"""character_prompts: a list of good working prompts per character (wanly-console#675)

Revision ID: 120
Revises: 119
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision = "120"
down_revision = "119"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "character_prompts",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("character_id", UUID(as_uuid=True),
                  sa.ForeignKey("ltx_characters.id", ondelete="CASCADE"), nullable=False),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_character_prompts_character_id", "character_prompts", ["character_id"])


def downgrade() -> None:
    op.drop_index("ix_character_prompts_character_id", table_name="character_prompts")
    op.drop_table("character_prompts")
