"""Add FaceFusion Final Cut columns: segments.facefusion_face_index/distance

Revision ID: 034
Revises: 033
Create Date: 2026-07-07

Final Cut's stage-2 engine moves from Wan Animate to FaceFusion. A facefusion
segment (reprocess_type="facefusion") stores which face to swap in a multi-person
clip (facefusion_face_index, left-to-right) and an optional reference-face-distance
override (facefusion_distance; NULL -> daemon auto: 1.0 single / 0.6 targeted).
Both nullable / server-defaulted so existing rows backfill cleanly.
"""

import sqlalchemy as sa
from alembic import op

revision = "034"
down_revision = "033"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "segments",
        sa.Column("facefusion_face_index", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column("segments", sa.Column("facefusion_distance", sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("segments", "facefusion_distance")
    op.drop_column("segments", "facefusion_face_index")
