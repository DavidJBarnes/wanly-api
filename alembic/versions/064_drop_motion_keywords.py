"""Drop segments.motion_keywords

Revision ID: 064
Revises: 063
Create Date: 2026-08-13

These were never measured. The daemon derived them by substring-matching the PROMPT against a
fixed table -- the function took the video bytes and ignored them -- and the table was loose
enough that "move" read as dancing, "pace" as walking, "hand" as a wave. A clip of none of those
things was logged as "Motion detected: walking, dancing, falling".

The column mattered because it did not stay still: this API handed it back as
previous_motion_keywords when the next segment was claimed, and the daemon appended canned
phrases from it and REPLACED the prompt before building the workflow. Removing the injection
(wanly-gpu-daemon#141) leaves this column with no reader and no writer, holding values that only
ever described the prompt they were parsed from.

Dropped rather than kept, because keeping it invites reading it as evidence of what a clip
contains, which is the one thing it never was. motion_magnitude stays -- that is real optical
flow over actual frames.

Irreversible in practice: the downgrade restores the column but not its contents, which is the
correct trade here since the contents were noise.
"""
import sqlalchemy as sa
from alembic import op

revision = "064"
down_revision = "063"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_column("segments", "motion_keywords")


def downgrade() -> None:
    op.add_column("segments", sa.Column("motion_keywords", sa.JSON(), nullable=True))
