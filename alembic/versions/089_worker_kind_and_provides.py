"""What a worker IS, and what it runs

Revision ID: 089
Revises: 088
Create Date: 2026-09-06

wanly-api#269. The workers table has only ever described render workers -- comfyui_running,
gpu_stats, checkpoints, loras, fetchable_kinds are all render concepts -- because until
wanly-services existed there was only one kind of worker.

TWO COLUMNS, because they answer different questions and one of them is load-bearing for the
claim path:

  kind      may I hand this work? Scalar, closed set, read by the claim gate and by
            queue-health. NOT NULL with a default, so there is never a live worker whose
            claimability is unknown -- and so every existing row and every current daemon
            keeps exactly its present meaning without a migration guessing at it.

  provides  what is this box actually doing? A list, because a services container runs
            several at once (SERVICES=joycaption,qwen-edit). Nullable, where NULL means
            "never reported" and [] means "reports nothing" -- the same distinction
            checkpoints, loras and fetchable_kinds already make, and for the same reason: an
            older daemon must not be indistinguishable from one with nothing to offer.

WHY NOT ONE FIELD. Keying the gate on a name allowlist ("is ltx-engine in provides?") fails in
the dangerous direction: add an engine, forget the allowlist, and that worker claims nothing --
which this codebase already knows looks exactly like an empty queue. With kind, a new engine is
`render` and claimable by default while a new service is `service` and excluded by default.
Both defaults land on the safe side.

provides also records something the API has never known: there is no `engine` column, so
nothing here has ever said whether a worker runs LTX or WAN 2.2. It was inferable while there
was one engine; it stops being inferable the moment a second kind of worker exists.
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "089"
down_revision = "088"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # server_default as well as a model default: the backfill of existing rows and any INSERT
    # from a connection that has not seen the new model both have to land on 'render'. A
    # nullable kind would mean a window where a live worker's claimability is unknown, and the
    # claim gate would have to guess -- which is the whole thing this column exists to stop.
    op.add_column(
        "workers",
        sa.Column("kind", sa.String(length=20), nullable=False, server_default="render"),
    )
    op.add_column(
        "workers",
        sa.Column("provides", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("workers", "provides")
    op.drop_column("workers", "kind")
