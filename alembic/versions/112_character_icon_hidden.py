"""Characters: an icon, and hidden (wanly-api#404)

The Characters redesign (wanly-console#616/#617):

  icon_uri   the one image that stands for a character wherever it is shown or picked.
             NULL falls back, in the console, to image_uri / face_ref_uri / sheet_uri.
  hidden     left out of every picker. Not deleted and not disabled: it keeps its runs, its
             LoRA and any job that already uses it, and still renders.

Both additive; no data changes. hidden defaults false, so every existing character stays
offered.

Revision ID: 112
Revises: 111
Create Date: 2026-10-05
"""
import sqlalchemy as sa
from alembic import op

revision = "112"
down_revision = "111"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("ltx_characters", sa.Column("icon_uri", sa.Text(), nullable=True))
    op.add_column("ltx_characters", sa.Column("hidden", sa.Boolean(), nullable=False,
                                              server_default=sa.text("false")))


def downgrade() -> None:
    op.drop_column("ltx_characters", "hidden")
    op.drop_column("ltx_characters", "icon_uri")
