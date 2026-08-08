"""Let a worker say which RunPod pod it is

Revision ID: 059
Revises: 058
Create Date: 2026-08-08

The console pairs RunPod pods with registered workers so a pod that is still booting shows as
"Starting" and stops showing once its worker appears. That pairing was done on name, which only
holds for pods launched through the console launcher — it sets the pod name and FRIENDLY_NAME to
the same value.

A pod launched from the RunPod template gets an auto-generated name (valid_chocolate_cockroach)
while its worker registers as runpod-<pod id>. Those never match, so the pod showed as "Starting"
forever alongside the worker it had already become.

The worker knows its own pod id — RunPod injects RUNPOD_POD_ID and the daemon already reads it
for self-termination. Reporting it removes the guessing entirely.
"""
import sqlalchemy as sa
from alembic import op

revision = "059"
down_revision = "058"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("workers", sa.Column("runpod_pod_id", sa.String(length=64), nullable=True))


def downgrade() -> None:
    op.drop_column("workers", "runpod_pod_id")
