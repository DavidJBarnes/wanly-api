"""Re-roll until: a take can carry the rule that judges it

Revision ID: 068
Revises: 067
Create Date: 2026-08-14

Re-rolling (065) made "show me another take" a button. This makes it a loop: when the user
re-rolls they can attach a rule — one of the four quality axes and a threshold — and when the
take finishes generating, the API judges it against that rule. A take that misses the rule is
archived and re-rolled again automatically, up to a global per-job cap (app setting
"max_rerolls_per_job", default 3). A take that meets the rule, or exhausts the cap, sits and
waits for the user exactly as before.

The rule lives ON THE SEGMENT, not the job, for the same reason the seed does: each take is
judged by the rule it was generated under. Changing the rule mid-chain must not relabel the
takes already archived, and an archived take's record should say what bar it was asked to clear.

reroll_count is the take's position in its chain — 1 for the user-initiated roll, incremented by
each automatic one — so the cap is enforced from the take itself rather than by counting
archived rows, which would conflate rule-driven rolls with plain manual ones.

All three columns are nullable and nothing is backfilled: NULL means "no rule", which is every
take that exists today and every plain re-roll from now on.
"""
import sqlalchemy as sa
from alembic import op

revision = "068"
down_revision = "067"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("segments", sa.Column("reroll_rule_metric", sa.String(length=16), nullable=True))
    op.add_column("segments", sa.Column("reroll_rule_threshold", sa.Float(), nullable=True))
    op.add_column("segments", sa.Column("reroll_count", sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column("segments", "reroll_count")
    op.drop_column("segments", "reroll_rule_threshold")
    op.drop_column("segments", "reroll_rule_metric")
