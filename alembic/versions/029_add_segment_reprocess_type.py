"""Add reprocess_type to segments

Revision ID: 029
Revises: 028
Create Date: 2026-05-13
"""

import sqlalchemy as sa
from alembic import op

revision = "029"
down_revision = "028"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("segments", sa.Column("reprocess_type", sa.String(20), nullable=True))


def downgrade() -> None:
    op.drop_column("segments", "reprocess_type")
