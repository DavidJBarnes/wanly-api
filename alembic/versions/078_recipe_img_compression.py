"""Per-pose img_compression override

Revision ID: 078
Revises: 077
Create Date: 2026-09-01

NULLABLE on purpose, and null means "use the stack's value" — the same shape as
`negative_prompt` and `frames`, which resolve as `r.frames or LTX_STACK["frames"]`. Every
existing pose keeps rendering at the stack default, so this migration changes no output.

img_compression is a video CRF: the conditioning frame is H.264-encoded at this quality and
decoded back before it anchors the render. 0 bypasses it. Measured at 18 (current) against 4
on a 4090, the lower value cut per-frame divergence by ~34% (wanly-gpu-docker#48) — which is
why it is worth exposing per pose rather than only globally.
"""
import sqlalchemy as sa
from alembic import op

revision = "078"
down_revision = "077"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("ltx_recipes", sa.Column("img_compression", sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column("ltx_recipes", "img_compression")
