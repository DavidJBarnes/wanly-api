"""Default pose and default character (wanly-console#543)

Revision ID: 105
Revises: 104
Create Date: 2026-09-28

The New Job and Next Segment modals open with nothing picked, and nearly every job is the
same pose on the same character. `is_default` marks the one of each the console preselects
when nothing else already is.

AT MOST ONE PER TABLE, enforced here rather than trusted to the route: a partial unique
index over the TRUE rows only. A plain unique on the column would allow one FALSE row too,
which is every row. The route clears the old default and sets the new one in one
transaction; the index is the backstop for two of those racing.

NOT NULL DEFAULT false, so every existing row is "not the default" and nothing is
preselected until someone stars one -- the modals behave exactly as before.
"""
import sqlalchemy as sa
from alembic import op

revision = "105"
down_revision = "104"
branch_labels = None
depends_on = None


def upgrade() -> None:
    for table in ("ltx_recipes", "ltx_characters"):
        op.add_column(table, sa.Column("is_default", sa.Boolean(), nullable=False,
                                       server_default=sa.false()))
        op.create_index(f"uq_{table}_one_default", table, ["is_default"], unique=True,
                        postgresql_where=sa.text("is_default"))


def downgrade() -> None:
    for table in ("ltx_characters", "ltx_recipes"):
        op.drop_index(f"uq_{table}_one_default", table_name=table)
        op.drop_column(table, "is_default")
