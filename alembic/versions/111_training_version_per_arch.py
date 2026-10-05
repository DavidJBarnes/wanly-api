"""Training: one live run per character+version PER ARCH (wanly-api#402)

KimJule's SDXL v1 was training and her LTX v1 could not be queued: "KimJule v1 is already
running". An SDXL LoRA and an LTX LoRA are different models, not versions of each other, and
nothing downstream can mix them -- the trainer's run dirs (sdxl-vN / ltx23b-vN) and the S3 keys
(character/sdxl/<stem>_sdxl_vN... / character/<stem>_vN...) were already apart. Only this index
put them in one sequence.

The index gains the arch, as `coalesce(config->>'arch', 'ltx')`: every row from before SDXL
has no arch and is LTX. Still partial on the live states, so finished runs never collide.

NO DATA IS CHANGED, and the new index is strictly looser than the old one, so creating it
cannot fail on existing rows. The DOWNGRADE can: put back, the old index refuses a character
with an SDXL and an LTX run of the same version both live. That is the state this migration
exists to allow; cancel one of them first.

Revision ID: 111
Revises: 110
Create Date: 2026-10-05
"""
from alembic import op

revision = "111"
down_revision = "110"
branch_labels = None
depends_on = None

LIVE = "status NOT IN ('completed','failed','cancelled')"


def upgrade() -> None:
    op.execute(
        "CREATE UNIQUE INDEX uq_training_jobs_character_version_arch_live ON training_jobs "
        f"(character, version, (coalesce(config->>'arch', 'ltx'))) WHERE {LIVE}")
    op.drop_index("uq_training_jobs_character_version_live", table_name="training_jobs")


def downgrade() -> None:
    op.execute(
        "CREATE UNIQUE INDEX uq_training_jobs_character_version_live ON training_jobs "
        f"(character, version) WHERE {LIVE}")
    op.drop_index("uq_training_jobs_character_version_arch_live", table_name="training_jobs")
