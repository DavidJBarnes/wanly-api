"""Add identity_reference_image to jobs

Revision ID: 046
Revises: 045
Create Date: 2026-07-13

Canonical VACE identity reference (a face crop from seg0) fed to every downstream segment's
VACE ref_images so identity re-anchors at each continuation instead of drifting.
"""
import sqlalchemy as sa
from alembic import op

revision = "046"
down_revision = "045"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("jobs", sa.Column("identity_reference_image", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("jobs", "identity_reference_image")
