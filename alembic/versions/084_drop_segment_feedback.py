"""Drop segment rating, notes and observation tags

Revision ID: 084
Revises: 083
Create Date: 2026-09-04

The observations modal is gone (console#385) and with it the only UI that ever wrote these
three columns. David decided the whole modal rather than the tags alone: rating, notes and
observation tags were one judgement recorded in one PATCH, and splitting them would leave a
form that captures two thirds of an opinion.

## Why they existed, and why that stopped being true

They were primary evidence. The metrics could not rank quality -- expression rewarded the
mouth-gape artifact it should have penalised -- so what a person saw outranked what was
measured. The metrics they existed to correct were themselves removed in #151 and #074, and
the vocabulary went with the experiments it was labelling ground truth for.

## Nothing is destroyed unchecked

Production carries no feedback at all: 0 of 264 segments have a rating, a note or a tag as of
2026-09-04 -- 073 purged the WAN history these were recorded against. But this migration also
runs against development databases and restored backups, and a rating is human judgement that
exists nowhere else. So anything present is COPIED FIRST, in the spirit of `purged_s3_paths`
in 073: `purged_segment_feedback` keeps the segment id and the three values, and the drop
proceeds. On production that table is created empty and costs nothing; anywhere else the
judgement survives the schema that held it.

The table is deliberately NOT dropped on downgrade -- it is the only copy.
"""
from alembic import op
import sqlalchemy as sa


revision = "084"
down_revision = "083"
branch_labels = None
depends_on = None


def upgrade():
    # IF NOT EXISTS because downgrade deliberately leaves this table behind -- it is the only
    # copy of the judgement. Without this a downgrade followed by an upgrade fails on
    # DuplicateTableError, which is a state anyone stepping a migration back and forth hits.
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS purged_segment_feedback (
            segment_id       UUID PRIMARY KEY,
            notes            TEXT,
            rating           SMALLINT,
            observation_tags VARCHAR(500),
            purged_at        TIMESTAMPTZ DEFAULT now()
        )
        """
    )
    # Only rows that actually carry something. An all-NULL row is not a lost opinion.
    #
    # ON CONFLICT refreshes rather than skips: on a re-upgrade the columns have just been
    # repopulated BY the downgrade, and if they were edited in between, the newer value is
    # the one worth keeping.
    op.execute(
        """
        INSERT INTO purged_segment_feedback (segment_id, notes, rating, observation_tags)
        SELECT id, notes, rating, observation_tags FROM segments
        WHERE rating IS NOT NULL
           OR nullif(btrim(coalesce(notes, '')), '') IS NOT NULL
           OR nullif(btrim(coalesce(observation_tags, '')), '') IS NOT NULL
        ON CONFLICT (segment_id) DO UPDATE
            SET notes            = EXCLUDED.notes,
                rating           = EXCLUDED.rating,
                observation_tags = EXCLUDED.observation_tags
        """
    )

    op.drop_column("segments", "observation_tags")
    op.drop_column("segments", "rating")
    op.drop_column("segments", "notes")


def downgrade():
    op.add_column("segments", sa.Column("notes", sa.Text(), nullable=True))
    op.add_column("segments", sa.Column("rating", sa.SmallInteger(), nullable=True))
    op.add_column("segments", sa.Column("observation_tags", sa.String(500), nullable=True))
    # Put back whatever upgrade set aside. The table is left in place afterwards: it is the
    # only copy, and a downgrade followed by another upgrade must not find it missing.
    op.execute(
        """
        UPDATE segments s
           SET notes = p.notes, rating = p.rating, observation_tags = p.observation_tags
          FROM purged_segment_feedback p
         WHERE p.segment_id = s.id
        """
    )
