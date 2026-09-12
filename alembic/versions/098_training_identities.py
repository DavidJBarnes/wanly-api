"""Generalize TrainingJob.second_identity into an identities list

Revision ID: 098
Revises: 097
Create Date: 2026-09-12

#102 gave a joint run ONE second identity. #106 needs a THIRD group -- two-person frames
trained alongside the solo sets -- and a list is the honest shape for "one or more extra
groups" rather than a second nullable column that would need a fourth for the next case.

The backfill is the whole reason this is safe: every existing row with a second_identity
becomes a one-element list, byte-for-byte, before the old column is dropped. A row with
none stays NULL. Nothing is lost and no reader sees a half-migrated shape.
"""
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "098"
down_revision = "097"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "training_jobs",
        sa.Column("identities", postgresql.JSONB(), nullable=True),
    )
    # The group is made SELF-CONTAINED: #102 stored its caption separately in
    # config.second_caption, and the claim reads each group's own caption. Fold it in, with
    # the "<trigger>, <gender>" form as the fallback so an old row renders exactly as it did.
    op.execute(
        "UPDATE training_jobs SET identities = jsonb_build_array("
        "  second_identity || jsonb_build_object('caption', COALESCE("
        "    config->>'second_caption',"
        "    CASE WHEN second_identity->>'gender' IS NOT NULL"
        "         THEN (second_identity->>'trigger') || ', ' || (second_identity->>'gender')"
        "         ELSE second_identity->>'trigger' END)))"
        "WHERE second_identity IS NOT NULL"
    )
    op.drop_column("training_jobs", "second_identity")


def downgrade() -> None:
    op.add_column(
        "training_jobs",
        sa.Column("second_identity", postgresql.JSONB(), nullable=True),
    )
    # The first element is the second identity #102 meant; a run with more than two groups
    # cannot round-trip and says so by keeping only the first.
    op.execute(
        "UPDATE training_jobs SET second_identity = identities -> 0 "
        "WHERE identities IS NOT NULL AND jsonb_array_length(identities) > 0"
    )
    op.drop_column("training_jobs", "identities")
