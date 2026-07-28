"""Add Lynx identity-preserving engine columns

Revision ID: 050
Revises: 049
Create Date: 2026-07-28

Lynx (ByteDance) runs Wan2.1 T2V-14B with ID/Ref adapters that re-assert identity at
every denoising step. Job carries the tunables (all nullable -> daemon settings default,
matching the existing per-job-override precedence); Segment carries the identity QA
scores the daemon measures after each render.
"""
import sqlalchemy as sa
from alembic import op

revision = "050"
down_revision = "049"
branch_labels = None
depends_on = None

# (name, type) for the job-level Lynx tunables.
JOB_COLUMNS = [
    ("generation_engine", sa.String(20)),
    ("lynx_subject_image", sa.Text()),
    ("lynx_ip_scale", sa.Float()),
    ("lynx_ref_scale", sa.Float()),
    ("lynx_cfg_scale", sa.Float()),
    ("lynx_start_percent", sa.Float()),
    ("lynx_end_percent", sa.Float()),
    ("lynx_ref_blocks_to_use", sa.Text()),
    ("lynx_ip_layers", sa.Text()),
    ("lynx_resampler", sa.Text()),
    ("lynx_steps", sa.Integer()),
    ("lynx_cfg", sa.Float()),
    ("lynx_shift", sa.Float()),
    ("lynx_scheduler", sa.String(32)),
    ("lynx_distill_strength", sa.Float()),
]


def upgrade() -> None:
    for name, coltype in JOB_COLUMNS:
        op.add_column("jobs", sa.Column(name, coltype, nullable=True))
    op.add_column("segments", sa.Column("lynx_identity_scores", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("segments", "lynx_identity_scores")
    for name, _ in reversed(JOB_COLUMNS):
        op.drop_column("jobs", name)
