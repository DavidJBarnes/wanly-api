"""Standing requests to launch a worker when a GPU frees up

Revision ID: 058
Revises: 057
Create Date: 2026-08-07

Availability swings hard — the 4090 in EU-RO-1 was available in 10/10 samples one morning and
0/7 that evening. The gap between a pod freeing up and someone asking for it is what a poller
closes and a person does not.

The row exists because the alternative is holding the reservation in memory, and both places it
could live are wrong: the browser tab gets closed, and the API container is recreated on every
deploy. A reservation that dies on deploy dies invisibly, which is worse than not offering the
feature — the user is still waiting for a worker nobody is going to launch.
"""
import sqlalchemy as sa
from alembic import op

revision = "058"
down_revision = "057"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "gpu_reservations",
        sa.Column("id", sa.dialects.postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="pending"),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("drain_after_jobs", sa.Integer(), nullable=True),
        sa.Column("pod_id", sa.String(length=64), nullable=True),
        sa.Column("error", sa.String(length=500), nullable=True),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    # The poller only ever looks for pending rows; without this it scans every reservation ever
    # made, forever.
    op.create_index(
        "ix_gpu_reservations_pending",
        "gpu_reservations",
        ["status", "expires_at"],
        postgresql_where=sa.text("status = 'pending'"),
    )


def downgrade() -> None:
    op.drop_index("ix_gpu_reservations_pending", table_name="gpu_reservations")
    op.drop_table("gpu_reservations")
