"""A worker may be several kinds at once

Revision ID: 094
Revises: 093
Create Date: 2026-09-08

One container per GPU (wanly-gpu-docker#83) runs the render stack AND the trainer, and
registers once. `kind` stays -- every gate and the console read it -- and becomes the
first of `kinds`, with `render` always first when present so the render gates and the
queue-health count keep meaning what they mean. `kinds` is what the training gate reads
and what the Workers page lists.
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "094"
down_revision = "093"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("workers", sa.Column("kinds", JSONB(), nullable=True))
    op.execute("UPDATE workers SET kinds = jsonb_build_array(kind) WHERE kinds IS NULL")


def downgrade() -> None:
    op.drop_column("workers", "kinds")
