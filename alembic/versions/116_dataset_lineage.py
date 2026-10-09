"""datasets.derived: which image in a set was made from which (wanly-api#445)

Revision ID: 116
Revises: 115
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "116"
down_revision = "115"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("datasets", sa.Column("derived", postgresql.JSONB(), nullable=False,
                                        server_default=sa.text("'{}'::jsonb")))


def downgrade() -> None:
    op.drop_column("datasets", "derived")
