"""Record the seed on takes that were archived before seeds were recorded

Revision ID: 066
Revises: 065
Create Date: 2026-08-14

Re-roll stamps the outgoing take with the seed it ran on, but that only started when the endpoint
learned to do it. Takes archived before then still carry NULL, which means "derive it from the
job" -- a statement that was unambiguous when a job had one take and stops being so once it has
several. In the collapsed list of previous takes they show no seed at all, which is precisely the
question that view exists to answer.

The value is not a guess. Until a segment carried its own seed, the worker was handed
(job.seed + index) at claim time, so this writes down what actually generated those clips.

Live segments are deliberately left NULL. Their seed is still derived, and it is still
unambiguous while they hold their position -- and NULL there keeps meaning "never asked for a
particular seed", which is the honest record. They get stamped if and when they are archived.

The arithmetic runs in `numeric`, not `bigint`. A job seed can be within a few of 2**63-1 (95% of
them were drawn from the full range), so `job.seed + index` can overflow bigint and abort the
whole migration before the modulo ever brings it back into range. Postgres does not wrap on
overflow, it raises.
"""
from alembic import op

revision = "066"
down_revision = "065"
branch_labels = None
depends_on = None

# One source of truth for the expression, so the test can execute exactly what ships.
BACKFILL_SQL = """
UPDATE segments AS s
SET seed = ((j.seed::numeric + s.index) % 9223372036854775807)::bigint
FROM jobs AS j
WHERE j.id = s.job_id
  AND s.discarded
  AND s.seed IS NULL
"""


def upgrade() -> None:
    op.execute(BACKFILL_SQL)


def downgrade() -> None:
    # Not reversible: which archived takes were stamped by this and which were stamped on archive
    # is not recorded, and clearing both would lose provenance that was correct.
    pass
