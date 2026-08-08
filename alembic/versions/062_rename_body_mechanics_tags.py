"""Rename the body-mechanics observation tags onto the travel axis

Revision ID: 062
Revises: 061
Create Date: 2026-08-08

The tag went through three names in one day, which is worth recording because each rename
sharpened what was actually being judged:

  bodies-unison   described the behaviour without saying it was wrong, unlike every other tag
  bodies-rocking  used the reviewer's own word, but "rocking" is how the failure LOOKS from
                  outside, not what is missing
  bodies-locked   names the mechanism: the bodies stay joined and there is no relative travel

The axis is travel. He withdraws and re-enters, so the two bodies separate and rejoin along an
axis; the failure is that they move as one unit. Naming both ends for travel -- locked and
in-out -- keeps the judgement on the thing being judged, where "impact" named only the collision
at the end of a stroke rather than the displacement producing it.

Renamed in place rather than deprecated: five segments carry the old values and the labels exist
to be aggregated. Two spellings of one observation is exactly the failure a controlled
vocabulary is for. Substring replacement is safe here because no other tag contains these as a
prefix, and the replacements occupy the same positions in OBSERVATION_TAGS, so the stored
vocabulary ordering is preserved.
"""
import sqlalchemy as sa
from alembic import op

revision = "062"
down_revision = "061"
branch_labels = None
depends_on = None

_RENAMES = [("bodies-rocking", "bodies-locked"), ("bodies-impact", "bodies-inout")]


def _apply(pairs):
    for old, new in pairs:
        op.execute(
            sa.text(
                "UPDATE segments SET observation_tags = replace(observation_tags, :old, :new) "
                "WHERE observation_tags LIKE :pat"
            ).bindparams(old=old, new=new, pat=f"%{old}%")
        )


def upgrade() -> None:
    _apply(_RENAMES)


def downgrade() -> None:
    _apply([(new, old) for old, new in _RENAMES])
