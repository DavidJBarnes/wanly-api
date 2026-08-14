"""Move a re-rolled take's seed onto the job

Revision ID: 067
Revises: 066
Create Date: 2026-08-14

Re-roll used to write the new seed onto the segment and leave job.seed alone. That left a
re-rolled job showing TWO seeds — one on the job header and a different one on segment 0 — where
the job's was the seed of a take that had already been archived. Two numbers, and the more
prominent one wrong.

There is one live take, so there is one answer, and it belongs on the job: that is where the
header shows it and where the create dialog accepts it back to reproduce the job. Segment 0 then
derives from it like every other segment and needs no seed of its own.

So for each live segment carrying its own seed: move that seed to the job and clear it on the
segment. Four jobs on production, all re-rolls.

Nothing is lost. Migration 066 stamped every archived take with the seed it ran on, so the job
seed being replaced here is already recorded on the row that actually produced it. The EXISTS
guard enforces that rather than assuming it: a live segment with its own seed but no archived
sibling would be a shape this has not seen, and it is left alone instead of having the old job
seed silently dropped.
"""
from alembic import op

revision = "067"
down_revision = "066"
branch_labels = None
depends_on = None

MOVE_SEED_TO_JOB = """
UPDATE jobs AS j
SET seed = s.seed
FROM segments AS s
WHERE s.job_id = j.id
  AND NOT s.discarded
  AND s.seed IS NOT NULL
  AND EXISTS (
    SELECT 1 FROM segments AS archived
    WHERE archived.job_id = j.id
      AND archived.discarded
      AND archived.seed IS NOT NULL
  )
"""

CLEAR_SEED_ON_LIVE_SEGMENTS = """
UPDATE segments AS s
SET seed = NULL
FROM jobs AS j
WHERE s.job_id = j.id
  AND NOT s.discarded
  AND s.seed IS NOT NULL
  AND j.seed = s.seed
"""


def upgrade() -> None:
    # Order matters: the clear keys on the job already holding the value, so a failure between
    # the two leaves the seed recorded twice rather than nowhere.
    op.execute(MOVE_SEED_TO_JOB)
    op.execute(CLEAR_SEED_ON_LIVE_SEGMENTS)


def downgrade() -> None:
    # Not reversible: which job seeds came from a live segment is not recorded, and the previous
    # job seed lives on an archived take that this did not touch.
    pass
