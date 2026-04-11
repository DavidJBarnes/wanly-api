"""Add workers table (merged from wanly-gpu-registry)

Revision ID: 022
Revises: 021
Create Date: 2026-04-11
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "022"
down_revision = "021"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "workers",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("friendly_name", sa.String(255), nullable=False),
        sa.Column("hostname", sa.String(255), nullable=False),
        sa.Column("ip_address", sa.String(45), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="online-idle"),
        sa.Column("comfyui_running", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("last_heartbeat", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("registered_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("gpu_stats", JSONB, nullable=True),
        sa.Column("sd_scripts", JSONB, nullable=True),
        sa.Column("a1111", JSONB, nullable=True),
        sa.Column("drain_after_jobs", sa.Integer(), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("workers")
