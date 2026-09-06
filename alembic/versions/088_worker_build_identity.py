"""What code a worker is actually running

Revision ID: 088
Revises: 087
Create Date: 2026-09-06

Two workers ran different code for 14 hours and nothing said so. It surfaced as a 422 on a
content LoRA that looked random: a RunPod pod fetched it correctly, the 3090 could not,
because the pod's daemon was current and the 3090's was cloned before the fix existed.

NULLABLE, and null means "this daemon does not report it" -- an older daemon must keep
heartbeating rather than 422 itself out of the pool on upgrade day, which is how every
optional worker field before these behaves.

TWO FIELDS, NOT ONE, because there are two independent update channels and they drift
separately:

  daemon_commit  the daemon is git-cloned from main by start.sh at every container boot
  image_ref      start.sh, the downloader and the engine are baked into the image and only
                 change on pull + recreate

The 3090 demonstrated the difference: `docker restart` moved daemon_commit to current and
left image_ref 37 hours stale. One field could not have shown that.
"""
import sqlalchemy as sa
from alembic import op

revision = "088"
down_revision = "087"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("workers", sa.Column("daemon_commit", sa.Text(), nullable=True))
    op.add_column("workers", sa.Column("image_ref", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("workers", "image_ref")
    op.drop_column("workers", "daemon_commit")
