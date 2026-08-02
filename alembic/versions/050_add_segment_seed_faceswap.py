"""Add seed_faceswap to segments

Revision ID: 050
Revises: 049
Create Date: 2026-08-02

Seed re-anchor moves from a global app setting to a per-segment flag.

The global version could never fire: it gated on "a later segment exists at claim time",
but jobs are created with segment 0 only and continuations are appended afterwards, so
the successor never existed when segment 0 was claimed. Per-segment removes the gate
entirely — the flag is set by the author, and the face comes from the segment's own
faceswap selection instead of a hardcoded per-LoRA map.

The old app_settings row is deleted here; nothing reads it after this revision.
"""
import sqlalchemy as sa
from alembic import op

revision = "050"
down_revision = "049"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "segments",
        sa.Column("seed_faceswap", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.execute("DELETE FROM app_settings WHERE key = 'seed_faceswap'")


def downgrade() -> None:
    op.drop_column("segments", "seed_faceswap")
