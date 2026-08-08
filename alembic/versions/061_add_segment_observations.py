"""Record what a human saw in a segment

Revision ID: 061
Revises: 060
Create Date: 2026-08-08

The automated metrics cannot rank quality, and we know this rather than suspect it: on the one
controlled pair where both a measurement and a human judgement exist, the expression score ranked
them backwards (daemon#123). A gaping mouth is the largest landmark excursion a face can make, so
the metric scores the very artifact that makes a segment look worse.

So human judgement is the ranking channel, and until now it lived in a text file beside the app.
This gives it a home next to the numbers it has to be read against.

Three columns rather than one, because free text alone does not aggregate:

  notes             - what was seen, in words.
  rating            - 1-5 overall. Enough to rank the arms of a paired test, which is the
                      comparison these experiments are built around.
  observation_tags  - a controlled vocabulary, comma separated. This is the valuable one: tags
                      like "mouth-void" are LABELLED GROUND TRUTH, and a few dozen of them turn
                      fixing the expression metric from guesswork into a fitting problem -- we can
                      ask whether any computable signal predicts the label.

Nullable throughout and touched by nothing in generation: annotation must never be able to change
what a segment produces.
"""
import sqlalchemy as sa
from alembic import op

revision = "061"
down_revision = "060"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("segments", sa.Column("notes", sa.Text(), nullable=True))
    op.add_column("segments", sa.Column("rating", sa.SmallInteger(), nullable=True))
    op.add_column("segments", sa.Column("observation_tags", sa.String(length=500), nullable=True))


def downgrade() -> None:
    op.drop_column("segments", "observation_tags")
    op.drop_column("segments", "rating")
    op.drop_column("segments", "notes")
