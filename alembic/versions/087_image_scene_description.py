"""Cache an image's scene description on its row

Revision ID: 087
Revises: 086
Create Date: 2026-09-05

console#414. JoyCaption's description of a frame is expensive — 4.5s cold, 1.2s warm on the
2070, which also hosts Automatic1111 — and the same image starts many jobs. Until now every
one of them paid for it again: POST /captions/describe is deliberately stateless, so a
description the user accepted left no trace.

NOT A CACHE THAT CAN BE REBUILT
    The model is nondeterministic. Re-running it gives different words, and the words are
    what the person read and accepted before pressing go. That is why this is a stored
    value with its own "re-roll" action rather than a memoised computation, and why the
    tags endpoint may no longer delete a row just because its tags were cleared.

scene_instruction records HOW the frame was described. A caption written under "terse" and
one under "rich" are different artefacts; a row that holds one without saying which is a
value nobody can interpret later.
"""
import sqlalchemy as sa
from alembic import op

revision = "087"
down_revision = "086"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("image_meta", sa.Column("scene_description", sa.Text(), nullable=True))
    op.add_column("image_meta", sa.Column("scene_instruction", sa.Text(), nullable=True))
    op.add_column("image_meta",
                  sa.Column("scene_described_at", sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    op.drop_column("image_meta", "scene_described_at")
    op.drop_column("image_meta", "scene_instruction")
    op.drop_column("image_meta", "scene_description")
