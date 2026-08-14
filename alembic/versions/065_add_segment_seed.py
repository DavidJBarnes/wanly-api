"""Give a segment a seed of its own

Revision ID: 065
Revises: 064
Create Date: 2026-08-14

Until now the seed lived only on the job. Every segment's seed was derived at claim time as
(job.seed + index), so segment 0 was exactly job.seed and the entire job reproduced from that one
number. That is a good property and this migration does not take it away: the new column is
NULLABLE, nothing is backfilled, and a NULL seed still means "derive it from the job". Every
existing row keeps the seed it always had.

The column exists for the case the derivation cannot express: re-rolling a segment to see another
take of the same prompt. The seed has to change while the position does not, and the only way to
do that with a job-level seed is to overwrite job.seed -- which silently rewrites history. The
archived take keeps its video, its rating and its notes, while the number that actually produced
it is replaced by the number that produced its replacement. Every archived clip would then be
labelled with the seed of whatever was rolled last.

That is the worst thing to lose in this particular system. Seed is the dominant variable in what a
take looks like -- expression especially, far more than the LoRA is -- so "which seed gave me
that one?" is the question the archive exists to answer. A re-roll feature that destroys the
answer while producing more takes to compare would be worse than not having it.

Deliberately not made NOT NULL with a backfill. Writing (job.seed + index) into every existing row
would look equivalent and would not be: it would freeze today's derivation into historical data,
so a later change to how seeds are derived would silently disagree with rows that were never
explicitly seeded. NULL means "this segment never asked for a particular seed", which is the truth.
"""
import sqlalchemy as sa
from alembic import op

revision = "065"
down_revision = "064"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("segments", sa.Column("seed", sa.BigInteger(), nullable=True))


def downgrade() -> None:
    op.drop_column("segments", "seed")
