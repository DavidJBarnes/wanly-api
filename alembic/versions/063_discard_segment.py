"""Soft-delete a segment, keeping its feedback

Revision ID: 063
Revises: 062
Create Date: 2026-08-09

Deleting a segment destroyed the row, and with it the rating, tags and notes recorded against it.
That was tolerable when a segment was just output; it is not now that the annotations are the
primary evidence in every experiment -- a bad segment is often the MOST informative one, and
throwing away the observation to get it out of the video is exactly backwards.

The unique constraint on (job_id, index) becomes PARTIAL, covering only live rows. That is what
lets a discarded segment 2 and its replacement both be "segment 2": the discarded row keeps its
index so the record reads correctly, and the regenerated one takes the same position in the
video. Without this the replacement would have to be appended at the end and would play out of
order.
"""
import sqlalchemy as sa
from alembic import op

revision = "063"
down_revision = "062"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "segments",
        sa.Column("discarded", sa.Boolean(), nullable=False, server_default="false"),
    )
    op.drop_constraint("uq_segments_job_index", "segments", type_="unique")
    op.create_index(
        "uq_segments_job_index_live",
        "segments",
        ["job_id", "index"],
        unique=True,
        postgresql_where=sa.text("NOT discarded"),
    )


def downgrade() -> None:
    # Discarded rows must go before the full constraint can be restored, or duplicate indices
    # would block it. They are the only rows that can legally share an index.
    op.execute(sa.text("DELETE FROM segments WHERE discarded"))
    op.drop_index("uq_segments_job_index_live", table_name="segments")
    op.create_unique_constraint("uq_segments_job_index", "segments", ["job_id", "index"])
    op.drop_column("segments", "discarded")
