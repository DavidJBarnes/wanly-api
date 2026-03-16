"""Add transition column to segments

Revision ID: 018
Revises: 017
Create Date: 2026-03-16
"""

import sqlalchemy as sa
from alembic import op

revision = "018"
down_revision = "017"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("segments", sa.Column("transition", sa.String(20), nullable=True))


def downgrade() -> None:
    op.drop_column("segments", "transition")
