"""Delete every WAN 2.2 job and everything hanging off it

Revision ID: 073
Revises: 072
Create Date: 2026-08-31

WAN 2.2 is retired. Its history is not being kept: David, 2026-08-31, asked for it gone, and
the reasons are practical as well as preferential.

**Run-time estimates.** `get_estimation_rates` fits against completed segments in a rolling
30-day window. While WAN segments are in that window they define the pixel-law fallback, so a
new LTX resolution with no history of its own gets priced from a pipeline that no longer runs.
Clearing the history is what lets the fallback describe LTX (see the note in estimation.py).

**The measurements themselves.** WAN rates were fitted on a two-pass sampler with a high/low
model split, at durations LTX does not use, on segments whose post-processing included identity
scoring and motion analysis that no longer happen. Nothing about them transfers.

## What counts as WAN

A job with no segment carrying `ltx_recipe`. That is the discriminator rather than a date,
because it survives back-dating, re-runs and clock skew, and because it is the same field the
worker writes — a job the LTX path produced always has it.

LTX jobs and their ratings, tags and observations are untouched.

## Order, and why favourites come first

`segments` and `videos` are deleted explicitly rather than by cascade. The models declare
ON DELETE CASCADE but the database constraints do not have it — found by running this against
a seeded database, where it failed on `segments_job_id_fkey` and rolled back. `favorites` is a
different problem: it stores `item_type` + `item_ref`, a loose string with no foreign
key, so a favourited WAN video would survive as a row pointing at nothing. It has to be cleaned
BEFORE the videos it references are gone, because afterwards there is no way to tell which
refs were WAN.

## S3 is not deleted here, but its paths ARE recorded first

A migration deletes rows, not objects — but the rows are the only record of WHERE those objects
are. Delete them and every WAN video, last frame and start image becomes unreachable garbage:
still billed for, no longer findable, with nothing left to join against.

So every S3 path belonging to a doomed job is written to `purged_s3_paths` before its row goes.
A later pass deletes the objects and the table; until then nothing is lost and the storage is
at least accounted for.

Deliberately two steps. Rows are cheap to be wrong about, objects are not, and an emptied
bucket cannot be undone by a downgrade.

## Downgrade

There isn't one. Deleted rows are deleted. `downgrade` raises rather than pretending.
"""
import sqlalchemy as sa
from alembic import op

revision = "073"
down_revision = "072"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()

    # A WAN job is one where no segment carries an LTX recipe.
    wan = """
        SELECT j.id FROM jobs j
        WHERE NOT EXISTS (
            SELECT 1 FROM segments s
            WHERE s.job_id = j.id AND s.ltx_recipe IS NOT NULL
        )
    """

    before = conn.execute(sa.text("SELECT count(*) FROM jobs")).scalar()
    doomed = conn.execute(sa.text(f"SELECT count(*) FROM ({wan}) x")).scalar()

    # Record the S3 paths BEFORE the rows that name them are deleted. Without this the objects
    # survive in their buckets with nothing left to identify them by — billed for, unreachable,
    # and impossible to clean up afterwards.
    op.create_table(
        "purged_s3_paths",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("path", sa.Text(), nullable=False),
        sa.Column("source", sa.String(length=32), nullable=False),
        sa.Column("recorded_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("now()")),
    )
    for source, sql in (
        ("video", f"SELECT output_path FROM videos WHERE job_id IN ({wan})"),
        ("segment", f"SELECT output_path FROM segments WHERE job_id IN ({wan})"),
        ("last_frame", f"SELECT last_frame_path FROM segments WHERE job_id IN ({wan})"),
        ("start_image", f"SELECT starting_image FROM jobs WHERE id IN ({wan})"),
    ):
        conn.execute(sa.text(
            f"INSERT INTO purged_s3_paths (path, source) "
            f"SELECT p, :src FROM ({sql}) AS t(p) WHERE p IS NOT NULL AND p <> ''"
        ), {"src": source})
    recorded = conn.execute(sa.text("SELECT count(*) FROM purged_s3_paths")).scalar()

    # Favourites first: no FK, so they would otherwise survive pointing at nothing, and once
    # the videos are gone there is no way to identify which refs were WAN.
    conn.execute(sa.text(f"""
        DELETE FROM favorites f
        WHERE f.item_ref IN (
                SELECT v.output_path FROM videos v
                WHERE v.job_id IN ({wan}) AND v.output_path IS NOT NULL
            )
           OR f.item_ref IN (
                SELECT s.output_path FROM segments s
                WHERE s.job_id IN ({wan}) AND s.output_path IS NOT NULL
            )
    """))

    # Explicitly, in foreign-key order. The MODELS declare ondelete="CASCADE" on both
    # segments.job_id and videos.job_id, but the database constraints do not have it —
    # `confdeltype` is 'a' (NO ACTION) on both, because the migrations that created them
    # predate the model annotation and SQLAlchemy metadata does not reach into an existing
    # constraint.
    #
    # Found by running this migration against a seeded database: it failed outright with
    # ForeignKeyViolationError on segments_job_id_fkey and rolled back. Relying on the model
    # here would have shipped a migration that cannot run.
    conn.execute(sa.text(f"DELETE FROM videos WHERE job_id IN ({wan})"))
    conn.execute(sa.text(f"DELETE FROM segments WHERE job_id IN ({wan})"))
    conn.execute(sa.text(f"DELETE FROM jobs WHERE id IN ({wan})"))

    after = conn.execute(sa.text("SELECT count(*) FROM jobs")).scalar()
    print(f"  purged {doomed} WAN jobs ({before} -> {after}); {after} LTX jobs kept; "
          f"{recorded} S3 paths recorded in purged_s3_paths for a later cleanup pass",
          flush=True)


def downgrade() -> None:
    raise RuntimeError(
        "073 deleted WAN job history. There is nothing to restore — a downgrade that "
        "silently recreated an empty schema would be worse than failing. purged_s3_paths "
        "is deliberately left in place: it is the only remaining record of which objects "
        "those jobs owned."
    )
