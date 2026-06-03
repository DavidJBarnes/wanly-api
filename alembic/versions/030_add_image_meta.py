"""Add image_meta table for image tags

Revision ID: 030
Revises: 029
Create Date: 2026-06-03
"""

import sqlalchemy as sa
from alembic import op

revision = "030"
down_revision = "029"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "image_meta",
        sa.Column("path", sa.Text(), nullable=False),
        sa.Column("tags", sa.Text(), nullable=True),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("path"),
    )


def downgrade() -> None:
    op.drop_table("image_meta")
