"""Drop ltx_recipes.validated

Revision ID: 101
Revises: 100
Create Date: 2026-09-15

The last of the "is this pose proven" flag. It began in migration 071 as a column on a recipe,
moved meaning in 072 to mean the POSE was proven, and was never used: David does not set it,
so every pose carries the seeded default and the console's "validated" chip and "unvalidated
pose" warnings point at a value nobody maintains.

A boolean nobody maintains is worse than no boolean. The console read it as a quality signal
("unvalidated pose" on a dialog chip), which is a claim the data cannot support — and the
automated metrics have picked the wrong clip before, so the flag was never that anyway.

Non-destructive in the sense that matters: it holds a human opinion that was never recorded.
Dropping it removes a column, not information.
"""
import sqlalchemy as sa
from alembic import op

revision = "101"
down_revision = "100"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_column("ltx_recipes", "validated")


def downgrade() -> None:
    op.add_column(
        "ltx_recipes",
        sa.Column("validated", sa.Boolean(), nullable=False, server_default="false"),
    )
