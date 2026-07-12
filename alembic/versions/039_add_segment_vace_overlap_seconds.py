"""Add segments.vace_overlap_seconds

Revision ID: 039
Revises: 038
Create Date: 2026-07-12

Length (seconds) of the reconstructed lead-in a VACE-continuation segment carries.
Stitch trims this off the previous segment's tail so the reconstruction replaces it
seamlessly. NULL for traditional (non-VACE) segments. Part of the VACE seam fix
(moving the overlap trim out of the daemon and into stitch).
"""
import sqlalchemy as sa
from alembic import op

revision = "039"
down_revision = "038"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("segments", sa.Column("vace_overlap_seconds", sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("segments", "vace_overlap_seconds")
