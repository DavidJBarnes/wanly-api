"""Add per-segment identity scores

Revision ID: 053
Revises: 052
Create Date: 2026-08-03

Every segment is now scored for facial identity as it finishes generating, inline where
motion_magnitude is already measured. Stored on the segment rather than in a separate table
because it is a property of that generation, exactly like motion_magnitude.

Two means, deliberately not collapsed into one:
  identity_mean_cos      vs the START FRAME      - how far this generation drifted
  identity_mean_cos_ref  vs the identity ref     - is it the character at all
A low mean with a flat slope is a weak-identity problem (dataset/LoRA); a good mean with a
steep slope is temporal drift. The fixes differ, so the distinction has to survive to the UI.
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "053"
down_revision = "052"
branch_labels = None
depends_on = None

_COLUMNS = (
    ("identity_mean_cos", sa.Float()),
    ("identity_mean_cos_ref", sa.Float()),
    ("identity_min_cos", sa.Float()),
    ("identity_slope", sa.Float()),
    ("identity_frames", sa.Integer()),
    ("identity_no_face", sa.Integer()),
    ("identity_face_px_p50", sa.Float()),
    ("identity_yaw_max", sa.Float()),
)


def upgrade() -> None:
    for name, type_ in _COLUMNS:
        op.add_column("segments", sa.Column(name, type_, nullable=True))
    # per-yaw bands, per-frame series, stride, multi-face count
    op.add_column("segments", sa.Column("identity_metrics", postgresql.JSONB(), nullable=True))


def downgrade() -> None:
    op.drop_column("segments", "identity_metrics")
    for name, _ in reversed(_COLUMNS):
        op.drop_column("segments", name)
