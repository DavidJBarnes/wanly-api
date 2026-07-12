"""Add segments hologram columns

Revision ID: 040
Revises: 039
Create Date: 2026-07-12

AR hologram (Tier-0) support. A finalized job's index-0 segment is reused as the
queue carrier for a `reprocess_type="ar_hologram"` job: two resolved params
(key color, subject height) drive the daemon matte + manifest, and three output
paths hold the packed color+alpha mp4, the hologram.json manifest, and a
transparent poster PNG. All nullable — traditional segments are unaffected.
"""
import sqlalchemy as sa
from alembic import op

revision = "040"
down_revision = "039"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("segments", sa.Column("hologram_key_color", sa.String(length=20), nullable=True))
    op.add_column("segments", sa.Column("hologram_subject_height_m", sa.Float(), nullable=True))
    op.add_column("segments", sa.Column("hologram_video_path", sa.Text(), nullable=True))
    op.add_column("segments", sa.Column("hologram_manifest_path", sa.Text(), nullable=True))
    op.add_column("segments", sa.Column("hologram_poster_path", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("segments", "hologram_poster_path")
    op.drop_column("segments", "hologram_manifest_path")
    op.drop_column("segments", "hologram_video_path")
    op.drop_column("segments", "hologram_subject_height_m")
    op.drop_column("segments", "hologram_key_color")
