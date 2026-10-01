"""Character sheets: one-photo provenance (wanly-console#585)

Revision ID: 109
Revises: 108
Create Date: 2026-10-01

Build sheet now takes ONE photo of the person (face + body): it is image 1 of the Qwen
turnaround, so her build carries into all three views, and the sheet's face panel is
auto-cropped from that same photo. Migration 108's face_uri therefore holds that one photo, and
two things need a place:
  * photo_mode -- "one_photo" for these sheets. NULL is a sheet from before #585, built from a
    face photo plus body words (its face panel was a face-detected strip of that face photo);
  * face_panel_crop -- how the panel was cut from the photo: {"source": "same_photo", box,
    crop, padding, scale, detector, det_size, photo_size}.

ADD COLUMN ONLY, both nullable with no default: no existing character_sheets row is read or
changed, and every row written before this keeps exactly what it recorded.
"""
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "109"
down_revision = "108"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("character_sheets", sa.Column("photo_mode", sa.String(16), nullable=True))
    op.add_column("character_sheets",
                  sa.Column("face_panel_crop", postgresql.JSONB(), nullable=True))


def downgrade() -> None:
    op.drop_column("character_sheets", "face_panel_crop")
    op.drop_column("character_sheets", "photo_mode")
