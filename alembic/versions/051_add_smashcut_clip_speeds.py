"""Add per-clip playback speed to smashcut carriers

Revision ID: 051
Revises: 050
Create Date: 2026-07-31

Smashcut clips can now be retimed individually (slow-motion through fast-forward). The
speeds ride in a list parallel to the existing smashcut_clip_paths rather than reshaping
that column into objects, so a daemon running older code still reads the paths it expects
and simply builds the montage at 1x instead of failing on an unrecognised shape.

Nullable with no backfill: NULL means "no retiming", which is exactly what every existing
smashcut carrier did.
"""
import sqlalchemy as sa
from alembic import op

revision = "051"
down_revision = "050"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("segments", sa.Column("smashcut_clip_speeds", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("segments", "smashcut_clip_speeds")
