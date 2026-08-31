"""Drop video presets, the LoRA library, and the WAN sampler settings they carried

Revision ID: 075
Revises: 074
Create Date: 2026-08-31

WAN 2.2 is retired. These are the last of its configuration surface.

## Video presets

A preset is a named bundle of WAN sampler settings — lightx2v strengths, cfg high/low,
steps_total, high_noise_steps, flow_shift. `lightx2v` is WAN's distillation LoRA, and the
high/low split exists because WAN samples in two passes with a different model in each. LTX has
neither. Its equivalent values live in one global stack, are identical for every render, and are
not chosen per job.

The same parameters are dropped from `jobs`, because the only thing that read them was the claim
endpoint's preset resolution, which goes with the presets.

## The LoRA library

Every entry is a PAIR of files, high-noise and low-noise, each with its own S3 URI and default
weight, because WAN samples in two passes. An LTX character LoRA is ONE file applied at two
strengths on one model. The pairing is not merely unused — it is a different thing wearing the
same name.

LTX does not use this table at all: a character carries its LoRA filename (ltx_characters), the
recipe carries it into the segment (segments.ltx_recipe), and the engine resolves it against
what is on disk.

`segments.loras` goes with it. It held resolved WAN entries — lora_id, high_file, high_weight,
low_file, low_weight — and nothing on the LTX path writes or reads it.

## App settings

cfg_high, cfg_low, lightx2v_strength_high/low, steps_total, high_noise_steps and flow_shift were
the fleet-wide defaults for the same WAN parameters. max_rerolls_per_job capped an automatic
re-roll chain that no longer exists (#151). continuation_mode and vace_overlap_frames configure
VACE, which the daemon already treats as unsupported. Rows are left in place rather than deleted:
they are inert once nothing reads them, and app_settings is a key-value table where a stale row
costs nothing.

## S3 is not touched

The LoRA files remain in their bucket. As with 073, deleting objects is a separate, deliberate
step — and unlike job media, these are the ONLY copy of those LoRAs. The daemon treated its local
directory as a cache and re-downloaded from S3.

## Downgrade

Restores the shape, empty. The presets and library rows are not recoverable.
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision = "075"
down_revision = "074"
branch_labels = None
depends_on = None

_JOB_WAN_COLUMNS = ["lightx2v_strength_high", "lightx2v_strength_low", "cfg_high", "cfg_low",
                    "steps_total", "high_noise_steps", "flow_shift"]


def upgrade() -> None:
    # FK-bearing columns first, then the tables they point at.
    op.drop_column("segments", "video_preset_id")
    op.drop_column("segments", "loras")
    op.drop_column("jobs", "video_preset_id")
    for name in _JOB_WAN_COLUMNS:
        op.drop_column("jobs", name)
    op.drop_table("video_settings_presets")
    op.drop_table("loras")


def downgrade() -> None:
    raise RuntimeError(
        "075 dropped the video-preset and LoRA-library tables. Their rows are gone, and "
        "recreating empty tables would look like a restore while leaving every job that "
        "referenced them pointing at nothing."
    )
