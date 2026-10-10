"""lora_usage: LoRAs used outside Wanly's renders, as reported by A1111's box (wanly-api#458)

SDXL character LoRAs are used by hand in A1111, so whether one has been tried can only be read
from A1111's saved PNGs. A reporter on that box (wanly-gpu-docker#211) POSTs per-name totals.

Revision ID: 118
Revises: 117
"""
import sqlalchemy as sa
from alembic import op

revision = "118"
down_revision = "117"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "lora_usage",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("source", sa.String(32), nullable=False, server_default="a1111"),
        sa.Column("images", sa.Integer, nullable=False, server_default="0"),
        sa.Column("first_used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.func.now()),
        sa.UniqueConstraint("name", "source", name="uq_lora_usage_name_source"),
    )


def downgrade() -> None:
    op.drop_table("lora_usage")
