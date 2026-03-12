"""Add app_settings table

Revision ID: 015
Revises: 014
Create Date: 2026-03-12
"""

from alembic import op
import sqlalchemy as sa

revision = "015"
down_revision = "014"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "app_settings",
        sa.Column("key", sa.String(255), primary_key=True),
        sa.Column("value", sa.Text(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True)),
    )
    # Seed defaults
    op.execute(
        "INSERT INTO app_settings (key, value) VALUES "
        "('cfg_high', '1'), "
        "('cfg_low', '1'), "
        "('lightx2v_strength_high', '2.0'), "
        "('lightx2v_strength_low', '1.0')"
    )


def downgrade() -> None:
    op.drop_table("app_settings")
