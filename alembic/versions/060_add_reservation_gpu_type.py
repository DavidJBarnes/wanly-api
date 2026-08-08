"""Let a reservation name which GPU it is waiting for

Revision ID: 060
Revises: 059
Create Date: 2026-08-08

A reservation polls unattended for up to twelve hours, so which GPU it waits for is not a
cosmetic choice. Community 4090s are frequently unplaceable — RunPod matches a host and then
answers "this machine does not have the resources", which is a fit failure rather than an empty
fleet — while a 3090 places immediately. Without this column every reservation waits for the
server default, so a user who would happily have taken a 3090 gets nothing and an expired
window.

NULL means "the server default", which is exactly what every reservation created before this
column was waiting for. So there is no backfill: writing today's default into old rows would
invent a decision the user never made, and would freeze those rows if the default changed.
"""
import sqlalchemy as sa
from alembic import op

revision = "060"
down_revision = "059"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "gpu_reservations",
        sa.Column("gpu_type_id", sa.String(length=64), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("gpu_reservations", "gpu_type_id")
