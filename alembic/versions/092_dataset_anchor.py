"""An anchor image per dataset

Revision ID: 092
Revises: 091
Create Date: 2026-09-08

One image in a set, nominated as the face every other image is scored against.

The reference machinery that already existed scored against a whole dataset's MEAN, and that is
unusable on the sets people actually have. A mean over a set that still contains two people is a
blend of both and separates neither; with no reference at all the crops are scored against their
own mean, which only shows they resemble each other. One picked face has no such ambiguity.

A URI rather than an index, because removing an image reorders the list -- an index would
silently come to mean a different photograph.

Nullable, and no backfill: a dataset without an anchor is the normal state until someone picks
one, and that is exactly what "no reference" already meant.
"""
import sqlalchemy as sa
from alembic import op

revision = "092"
down_revision = "091"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("datasets", sa.Column("anchor_uri", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("datasets", "anchor_uri")
