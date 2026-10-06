"""Training jobs: delete_requests (wanly-api#413)

Checkpoints can now be deleted one at a time, forever. The API deletes the S3 copy at once,
but the trainer's disk is only reachable through the trainer's own poll, so the label is
recorded here for it to act on -- the same shape, and the same pull, as publish_requests.

Additive; no data changes.

Revision ID: 113
Revises: 112
Create Date: 2026-10-06
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "113"
down_revision = "112"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("training_jobs", sa.Column("delete_requests", JSONB(), nullable=True))


def downgrade() -> None:
    op.drop_column("training_jobs", "delete_requests")
