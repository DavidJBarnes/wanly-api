"""Add per-segment face swapper model and pixel boost

Revision ID: 055
Revises: 054
Create Date: 2026-08-05

Both were hardcoded in the daemon (`inswapper_128` / `512x512`), which made the swap stage
untunable — and the swap stage is now where the remaining headroom is. A controlled sweep
showed nothing on the generation side moves identity: with NO LoRA at all the clip scores
mean 0.695, the same as with a purpose-trained character LoRA. Identity has to be injected
by the swap, so the swap needs knobs.

Two specific reasons these are the knobs that matter:

  face_swapper_model  inswapper_128 is 128px native. hyperswap_1c_256, simswap_256 and
                      hififace_256 are 256 native, and simswap_unofficial_512 is 512 — so
                      the crop is stretched half as far, or not at all. The ComfyUI node's
                      own default is hyperswap_1c_256; we were overriding it with the
                      older model.

  pixel_boost         512x512 on a ~100px face is close to a no-op that costs ~16x compute:
                      the crop is interpolated up from 100px and then pixel-unshuffled into
                      16 strided sub-images carrying no extra real signal. Only worth it
                      when the face genuinely has >128px of detail.

NULL means "use the daemon default", matching every other per-segment override.
"""
import sqlalchemy as sa
from alembic import op

revision = "055"
down_revision = "054"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("segments", sa.Column("faceswap_model", sa.String(64), nullable=True))
    op.add_column("segments", sa.Column("faceswap_pixel_boost", sa.String(16), nullable=True))


def downgrade() -> None:
    op.drop_column("segments", "faceswap_pixel_boost")
    op.drop_column("segments", "faceswap_model")
