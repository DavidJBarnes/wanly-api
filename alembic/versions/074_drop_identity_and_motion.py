"""Drop identity scoring, motion analysis and the re-roll rule

Revision ID: 074
Revises: 073
Create Date: 2026-08-31

The daemon stopped writing all of this in wanly-gpu-daemon#152. These are the columns nothing
writes any more.

Identity scoring and motion analysis existed to compensate for WAN 2.2 drifting — measure the
damage, then re-roll or re-anchor against the measurement. LTX holds identity from the character
LoRA and the start frame. Measured on a real 241-frame render they also cost 326s and 15-39s
against a 263s render, so post-processing outweighed the render it analysed.

The metrics were also never trustworthy. Expression is mean landmark displacement, so the
mouth-gape artifact it should penalise scores as MORE expression. Motion is whole-frame optical
flow, so two bodies rocking in unison outscore the correct mechanic — a 5-rated segment measured
0.545 against a 3-rated one at 1.162. The observation tags outlived them precisely because they
record correctness where the metrics recorded quantity.

## Re-roll survives; re-roll UNTIL does not

Rolling another take by hand is unaffected — it is a button and a new seed. What goes is the
RULE: a metric and a threshold judged on completion, which compared against the mean of an
identity or motion series. With nothing producing those series a rule is permanently
unevaluable, and the old code degraded to logging "sitting idle" rather than failing. reroll_count
goes with it because only the rule machinery ever set it.

## lynx_identity_scores

The same measurement under another name, written by a Lynx render and read by nothing that still
exists. Lynx itself is retired — the daemon raises on it — and retires separately.

## The data

These columns hold real measurements from completed WAN segments. Almost all of those rows were
deleted in 073; what survives is LTX renders, whose values were produced by the scoring this
removes. Nothing is losing history that describes a pipeline still in use.

## Downgrade

Restores the columns, empty. The schema comes back; the measurements do not.
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "074"
down_revision = "073"
branch_labels = None
depends_on = None

_SEGMENT_COLUMNS = [
    "motion_magnitude",
    "identity_mean_cos",
    "identity_mean_cos_ref",
    "identity_min_cos",
    "identity_slope",
    "identity_frames",
    "identity_no_face",
    "identity_face_px_p50",
    "identity_yaw_max",
    "identity_start_cos_ref",
    "identity_end_cos_ref",
    "identity_metrics",
    "lynx_identity_scores",
    "reroll_rule_metric",
    "reroll_rule_threshold",
    "reroll_count",
]


def upgrade() -> None:
    for name in _SEGMENT_COLUMNS:
        op.drop_column("segments", name)
    op.drop_column("jobs", "identity_reference_image")


def downgrade() -> None:
    op.add_column("jobs", sa.Column("identity_reference_image", sa.Text(), nullable=True))
    op.add_column("segments", sa.Column("reroll_count", sa.Integer(), nullable=True))
    op.add_column("segments", sa.Column("reroll_rule_threshold", sa.Float(), nullable=True))
    op.add_column("segments", sa.Column("reroll_rule_metric", sa.String(length=16), nullable=True))
    op.add_column("segments", sa.Column("lynx_identity_scores", sa.JSON(), nullable=True))
    op.add_column("segments", sa.Column("identity_metrics", JSONB(), nullable=True))
    for name in ("identity_end_cos_ref", "identity_start_cos_ref", "identity_yaw_max",
                 "identity_face_px_p50", "identity_min_cos", "identity_slope",
                 "identity_mean_cos_ref", "identity_mean_cos", "motion_magnitude"):
        op.add_column("segments", sa.Column(name, sa.Float(), nullable=True))
    op.add_column("segments", sa.Column("identity_no_face", sa.Integer(), nullable=True))
    op.add_column("segments", sa.Column("identity_frames", sa.Integer(), nullable=True))
