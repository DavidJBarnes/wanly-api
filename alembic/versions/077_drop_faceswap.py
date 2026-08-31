"""Drop faceswap

Revision ID: 077
Revises: 076
Create Date: 2026-08-31

Faceswap was WAN 2.2's identity mechanism: generate, then swap the face back onto every frame
as a post-pass, because WAN drifted away from the subject over a clip.

LTX gets identity from the character LoRA plus the start frame. Measured: neither produces the
subject alone (cos 0.008 for the keyframe with no LoRA, 0.186 for the LoRA with no keyframe);
together they do (0.577). The LoRA is a stabiliser holding an identity the keyframe supplies,
which is precisely the job faceswap was doing badly.

It was never free, either. inswapper rebuilds a face from a 512-dimension embedding, which cost
a measured ~18% of face detail — and seeding a continuation from a swapped frame compounded it:
233px to 130px across segment 0, 101 once it became conditioning, 79 by the end of segment 1.
That is the reason the continuation seed was taken BEFORE the swap, a workaround that now has
nothing to work around.

`seed_faceswap` goes with it. It re-anchored a continuation's last frame to the canonical face
before that frame seeded the next segment — a compensation for the compounding above.

## What this does NOT touch

reprocess_type stays. Faceswap was one of three things that rode it; the AR hologram and
smashcut carriers still do, and both are wanted.
"""
import sqlalchemy as sa
from alembic import op

revision = "077"
down_revision = "076"
branch_labels = None
depends_on = None

_COLUMNS = [
    ("faceswap_enabled", sa.Boolean(), "false"),
    ("faceswap_method", sa.String(length=20), None),
    ("faceswap_source_type", sa.String(length=20), None),
    ("faceswap_image", sa.Text(), None),
    ("faceswap_faces_order", sa.Text(), None),
    ("faceswap_faces_index", sa.Text(), None),
    ("faceswap_model", sa.String(length=64), None),
    ("faceswap_pixel_boost", sa.String(length=16), None),
    ("seed_faceswap", sa.Boolean(), "false"),
]


def upgrade() -> None:
    for name, _type, _default in _COLUMNS:
        op.drop_column("segments", name)


def downgrade() -> None:
    for name, type_, default in reversed(_COLUMNS):
        op.add_column(
            "segments",
            sa.Column(name, type_, nullable=default is None, server_default=default),
        )
