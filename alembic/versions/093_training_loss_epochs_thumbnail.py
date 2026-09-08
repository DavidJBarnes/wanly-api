"""Loss curve, per-epoch checkpoints, publish-on-demand, and a face for the LoRA

Revision ID: 093
Revises: 092
Create Date: 2026-09-08

Three things asked for after the first end-to-end run (wanly-console#464 follow-up):

  loss_log          [[step, avr_loss], ...] as the trainer reports it, so the console can draw it.
  epochs            [{label, step, loss}, ...] -- every checkpoint the run wrote, published or
                    not. Only the final one is uploaded by default; the others sit on the
                    trainer until asked for.
  publish_requests  [label, ...] the user asked to have uploaded after the fact.
  thumbnail_uri     the dataset's anchor image at creation, so the LoRA has a face.

  ltx_characters.image_uri  the same face, on the character, set when a run publishes.
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "093"
down_revision = "092"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("training_jobs", sa.Column("loss_log", JSONB(), nullable=True))
    op.add_column("training_jobs", sa.Column("epochs", JSONB(), nullable=True))
    op.add_column("training_jobs", sa.Column("publish_requests", JSONB(), nullable=True))
    op.add_column("training_jobs", sa.Column("thumbnail_uri", sa.Text(), nullable=True))
    op.add_column("ltx_characters", sa.Column("image_uri", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("ltx_characters", "image_uri")
    op.drop_column("training_jobs", "thumbnail_uri")
    op.drop_column("training_jobs", "publish_requests")
    op.drop_column("training_jobs", "epochs")
    op.drop_column("training_jobs", "loss_log")
