"""Add wildcards table and segment prompt_template column

Revision ID: 005
Revises: 004
Create Date: 2026-02-13
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

revision = "005"
down_revision = "004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "wildcards",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(255), unique=True, nullable=False),
        sa.Column("options", sa.JSON, nullable=False, server_default="[]"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.add_column("segments", sa.Column("prompt_template", sa.Text, nullable=True))


def downgrade() -> None:
    op.drop_column("segments", "prompt_template")
    op.drop_table("wildcards")
