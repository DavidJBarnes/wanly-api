"""Per-image face size on datasets: datasets.faces (wanly-api#432)

{uri: measurement} -- the largest face's height at training size, the image size, pose -- from
the face-crop service's /measure (wanly-gpu-docker#206). Feeds the per-image face-size badge,
the training preflight's small-face warning and "Fix small faces".

Additive, NOT NULL with an empty-object default, like captions and scores: a NULL here would be
the JSONB none-as-null trap waiting to happen, and every existing set simply has nothing
measured yet.

Revision ID: 115
Revises: 114
Create Date: 2026-10-08
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "115"
down_revision = "114"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("datasets", sa.Column("faces", JSONB(), nullable=False,
                                        server_default=sa.text("'{}'::jsonb")))


def downgrade() -> None:
    op.drop_column("datasets", "faces")
