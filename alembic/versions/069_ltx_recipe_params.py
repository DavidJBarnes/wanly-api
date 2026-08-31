"""LTX recipe parameters on a segment

Revision ID: 069
Revises: 068
Create Date: 2026-08-31

LTX 2.3 replaces WAN 2.2 as the engine underneath Wanly. Most of what a render needs is
already on the segment and means the same thing for either engine — prompt, negative_prompt,
seed, start_image, output_path — so this adds only what is genuinely new.

What is new is the RECIPE. On the LTX side a render is driven by a validated (character, pose)
configuration authored in a sheet, and that configuration reduced, across sixteen validated
recipes, to exactly two variables: the character LoRA and the prompt. Every other field —
checkpoint, content LoRA, distill, guidance, steps, frames, resolution, negative — had one
distinct value across all sixteen. So the useful thing to record per segment is which recipe
ran and which of its defaults were overridden, not a column per parameter.

Hence one JSONB column rather than a dozen nullable ones. The shape:

    {"recipe": "Missionary POV", "character": "k3lly2026",
     "char_lora": "k3lly2026_v2", "char_s1": 0.8, "char_s2": 1.5,
     "frames": 241, "graph_sha256": "85649768667ba700..."}

`graph_sha256` is the one field worth calling out. A recipe is value patches on a pinned
graph, so the hash of the RESOLVED graph detects any change to shared state that alters a
recipe, at no GPU cost. It is how that project caught a regenerated sheet silently overwriting
a hand-authored prompt, within minutes, on the day the check was written. Recording it against
the segment is what makes a render provably the configuration that was signed off, rather than
something that merely claims to be.

Nullable, nothing backfilled: NULL means "not an LTX recipe render", which is every WAN
segment that already exists.
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "069"
down_revision = "068"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("segments", sa.Column("ltx_recipe", JSONB(), nullable=True))


def downgrade() -> None:
    op.drop_column("segments", "ltx_recipe")
