"""Un-pin the poses that hold a verbatim copy of the stack's negative prompt

Revision ID: 086
Revises: 085
Create Date: 2026-09-05

console#430. The Settings "negative prompt" field had never once been used. Two things kept
it out: the recipe book resolved a pose's negative against `LTX_STACK['negative']` rather
than the setting, and the console's pose editor prefilled its box from that RESOLVED value
and saved it straight back. Opening a pose and pressing Save therefore turned "inherit the
default" into a hardcoded copy of the constant, silently.

It has already happened to every pose. Migration 071 seeded all of them NULL; production now
holds 16 rows whose negative_prompt is byte-identical to the constant -- md5
bdb2e8d67a0c43e89d5aeee1cce2a0e0, 195 characters, on all 16. Fixing only the resolver would
change nothing for them: a non-NULL override wins, and every row has one.

MATCHED ON EXACT TEXT, NOT BLANKET-CLEARED
    A pose whose negative was genuinely hand-written must survive. Only rows equal to the
    constant are cleared: those carry no information the constant does not already carry, so
    reading them as "never overridden" loses nothing, and it is what the row meant before an
    editor round-trip pinned it.

THE TEXT IS FROZEN HERE, not imported from app.ltx_stack. A migration describes the data as
it was on this date. If the stack constant is edited later this must still match the rows it
was written for, and an import would quietly stop matching them.

The downgrade writes the constant back into every row it could have cleared. It cannot tell
those from a pose that always inherited -- but under the OLD resolver both rendered with the
constant, so the rendered result is identical either way.
"""
import sqlalchemy as sa
from alembic import op

revision = "086"
down_revision = "085"
branch_labels = None
depends_on = None

_PINNED = (
    "static, still image, frozen, no motion, slideshow, identity change, different "
    "person, face distortion, warped anatomy, extra limbs, deformed hands, merged limbs, "
    "mangled body, blurry, low quality"
)


def upgrade() -> None:
    op.get_bind().execute(
        sa.text("UPDATE ltx_recipes SET negative_prompt = NULL "
                "WHERE negative_prompt = :pinned"),
        {"pinned": _PINNED},
    )


def downgrade() -> None:
    op.get_bind().execute(
        sa.text("UPDATE ltx_recipes SET negative_prompt = :pinned "
                "WHERE negative_prompt IS NULL"),
        {"pinned": _PINNED},
    )
