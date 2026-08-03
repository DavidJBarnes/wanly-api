"""Tests for per-segment identity scoring on the API side.

The failure mode this guards is silence. Identity scores are measurement-only — they never
gate a segment's status — so if a field is dropped between the daemon and the console,
nothing errors. The numbers simply never appear, and the feature looks like it "doesn't work"
without a single log line. That is exactly how the seed re-anchor stayed dead through 26 jobs.

Unit-level (no live DB), matching the rest of this suite.
"""

import ast
from pathlib import Path

import pytest

from app.models import Segment
from app.schemas.jobs import IdentityAggregate, JobDetailResponse
from app.schemas.segments import SegmentResponse, SegmentStatusUpdate

ROOT = Path(__file__).resolve().parents[1]
JOBS_ROUTE = ROOT / "app" / "routes" / "jobs.py"
SEGMENTS_ROUTE = ROOT / "app" / "routes" / "segments.py"

IDENTITY_FIELDS = [
    "identity_mean_cos",
    "identity_mean_cos_ref",
    "identity_min_cos",
    "identity_slope",
    "identity_frames",
    "identity_no_face",
    "identity_face_px_p50",
    "identity_yaw_max",
    "identity_metrics",
]


class TestFieldsExistEndToEnd:
    """daemon -> API -> DB -> console. A gap anywhere shows up as missing numbers, not an error."""

    @pytest.mark.parametrize("field", IDENTITY_FIELDS)
    def test_daemon_can_send_it(self, field):
        assert field in SegmentStatusUpdate.model_fields

    @pytest.mark.parametrize("field", IDENTITY_FIELDS)
    def test_model_persists_it(self, field):
        assert hasattr(Segment, field)

    @pytest.mark.parametrize("field", IDENTITY_FIELDS)
    def test_console_receives_it(self, field):
        assert field in SegmentResponse.model_fields

    def test_patch_handler_persists_every_field(self):
        """SegmentStatusUpdate is applied field-by-field, so a field can exist on the schema
        AND the model and still never be written."""
        src = SEGMENTS_ROUTE.read_text()
        for field in IDENTITY_FIELDS:
            assert f"segment.{field} = body.{field}" in src, (
                f"PATCH /segments/{{id}} never assigns {field}; the daemon would send it "
                f"and the API would silently discard it"
            )


class TestTwoMeansStaySeparate:
    """Collapsing 'drift from start' and 'is it the character' into one number destroys the
    distinction the feature exists to expose. On the CTRL clip these were 0.811 and 0.489."""

    def test_both_means_are_distinct_fields(self):
        assert "identity_mean_cos" in SegmentResponse.model_fields
        assert "identity_mean_cos_ref" in SegmentResponse.model_fields

    def test_aggregate_keeps_both(self):
        assert "mean_cos" in IdentityAggregate.model_fields
        assert "mean_cos_ref" in IdentityAggregate.model_fields

    def test_slope_is_reported_separately_from_the_mean(self):
        """A low mean + flat slope is a dataset problem; a good mean + steep slope is drift.
        Different fixes, so the UI must be able to tell them apart."""
        assert "identity_slope" in SegmentResponse.model_fields
        assert "slope" in IdentityAggregate.model_fields


class TestJobAggregate:

    def test_aggregate_is_on_the_detail_response(self):
        assert "identity" in JobDetailResponse.model_fields

    def test_every_detail_construction_passes_the_aggregate(self):
        """JobDetailResponse is hand-built field-by-field in the route handlers, so an
        omitted field is dropped silently by Pydantic — the video_preset_id 'Custom' bug."""
        tree = ast.parse(JOBS_ROUTE.read_text())
        calls = [
            {kw.arg for kw in n.keywords if kw.arg}
            for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            and n.func.id == "JobDetailResponse"
        ]
        assert len(calls) == 2, f"expected 2 JobDetailResponse sites, found {len(calls)}"
        for i, kwargs in enumerate(calls):
            assert "identity" in kwargs, (
                f"JobDetailResponse construction #{i} omits 'identity'; it would be dropped"
            )

    def test_aggregate_defaults_to_none_when_nothing_scored(self):
        """Older jobs predate scoring. They must render, not 500."""
        assert IdentityAggregate().mean_cos is None
        assert IdentityAggregate().scored_segments == 0


class TestAggregateMath:
    """The helper is pure, so exercise it directly with stand-in segments."""

    @staticmethod
    def _load_helper():
        tree = ast.parse(JOBS_ROUTE.read_text())
        fn = next(n for n in tree.body
                  if isinstance(n, ast.FunctionDef) and n.name == "_identity_aggregate")
        ns: dict = {"IdentityAggregate": IdentityAggregate}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "<agg>", "exec"), ns)
        return ns["_identity_aggregate"]

    @staticmethod
    def _seg(index, frames, mean=None, ref=None, slope=None, no_face=0,
             start_ref=None, end_ref=None):
        class S:
            pass
        s = S()
        s.index, s.identity_frames = index, frames
        s.identity_mean_cos, s.identity_mean_cos_ref = mean, ref
        s.identity_slope, s.identity_no_face = slope, no_face
        s.identity_start_cos_ref, s.identity_end_cos_ref = start_ref, end_ref
        return s

    def test_no_scored_segments_returns_none(self):
        agg = self._load_helper()
        assert agg([self._seg(0, 0)]) is None
        assert agg([]) is None

    def test_mean_is_frame_weighted_not_a_flat_average(self):
        """A 20-frame segment must not count the same as a 200-frame one."""
        agg = self._load_helper()
        r = agg([self._seg(0, 100, mean=0.9), self._seg(1, 900, mean=0.5)])
        # flat average would be 0.70; frame-weighted is 0.54
        assert r.mean_cos == pytest.approx(0.54, abs=1e-6)

    def test_reports_the_worst_drifting_segment(self):
        agg = self._load_helper()
        r = agg([
            self._seg(0, 50, mean=0.9, slope=-0.001),
            self._seg(1, 50, mean=0.8, slope=-0.009),
            self._seg(2, 50, mean=0.85, slope=-0.002),
        ])
        assert r.worst_segment_index == 1
        assert r.worst_segment_slope == pytest.approx(-0.009)

    def test_counts_frames_and_no_face_across_segments(self):
        agg = self._load_helper()
        r = agg([self._seg(0, 70, mean=0.8, no_face=3), self._seg(1, 30, mean=0.8, no_face=1)])
        assert r.frames == 100 and r.no_face == 4 and r.scored_segments == 2

    def test_partial_scores_do_not_break_the_aggregate(self):
        """A segment can have a mean but no slope (too few frames to regress)."""
        agg = self._load_helper()
        r = agg([self._seg(0, 50, mean=0.8, slope=None), self._seg(1, 50, mean=0.6, slope=-0.01)])
        assert r.mean_cos == pytest.approx(0.7, abs=1e-6)
        assert r.slope == pytest.approx(-0.01)


class TestCumulativeLowPointIsSurfaced:
    """The bug this class exists for.

    A real 2-segment job scored 0.764 and 0.785 "vs own start frame" — both green, both
    read as healthy — while segment 1 sat at 0.544 against the job's original image and was
    visibly a different person. Each segment is measured from its OWN first frame, and a
    continuation's first frame is the previous segment's already-drifted last frame. So
    per-segment scores can rise while cumulative identity collapses.

    The aggregate must therefore expose the cumulative LOW POINT, not just an average.
    """

    def test_aggregate_exposes_the_cumulative_low_point(self):
        assert "min_cos_ref" in IdentityAggregate.model_fields
        assert "min_cos_ref_segment_index" in IdentityAggregate.model_fields

    def test_low_point_is_found_even_when_the_average_looks_fine(self):
        agg = TestAggregateMath._load_helper()
        seg = TestAggregateMath._seg
        r = agg([
            seg(0, 177, mean=0.764, ref=0.764, slope=-0.0003),
            seg(1, 177, mean=0.785, ref=0.544, slope=-0.0004),
        ])
        # the per-segment means both look healthy and even improve
        assert r.mean_cos == pytest.approx(0.7745, abs=1e-3)
        # but the cumulative low point tells the truth, and names the segment
        assert r.min_cos_ref == pytest.approx(0.544)
        assert r.min_cos_ref_segment_index == 1

    def test_low_point_is_none_when_no_reference_was_scored(self):
        agg = TestAggregateMath._load_helper()
        seg = TestAggregateMath._seg
        r = agg([seg(0, 50, mean=0.8, ref=None)])
        assert r.min_cos_ref is None and r.min_cos_ref_segment_index is None


class TestJobTrajectory:
    """David's model: seg0's start frame is ground truth. Every segment is measured against
    it, and because a continuation begins where the previous ended, the endpoints chain:

        seg0   0.98 -> 0.85    loss 0.13
        seg1   0.85 -> 0.60    loss 0.25
        job    0.98 -> 0.60

    A mean over frames cannot express that - a clip sliding 0.95 -> 0.65 averages about the
    same as one sitting flat at 0.80, and only the first has lost the character.
    """

    def test_aggregate_exposes_the_job_trajectory(self):
        assert "start_cos_ref" in IdentityAggregate.model_fields
        assert "end_cos_ref" in IdentityAggregate.model_fields

    def test_trajectory_spans_first_segment_start_to_last_segment_end(self):
        agg = TestAggregateMath._load_helper()
        seg = TestAggregateMath._seg
        r = agg([
            seg(0, 177, mean=0.9, ref=0.9, start_ref=0.98, end_ref=0.85),
            seg(1, 177, mean=0.7, ref=0.7, start_ref=0.85, end_ref=0.60),
        ])
        assert r.start_cos_ref == pytest.approx(0.98)
        assert r.end_cos_ref == pytest.approx(0.60)

    def test_segments_are_ordered_by_index_not_list_order(self):
        """Segments can arrive in any order; the trajectory must still run 0 -> N."""
        agg = TestAggregateMath._load_helper()
        seg = TestAggregateMath._seg
        r = agg([
            seg(1, 50, mean=0.7, ref=0.7, start_ref=0.85, end_ref=0.60),
            seg(0, 50, mean=0.9, ref=0.9, start_ref=0.98, end_ref=0.85),
        ])
        assert r.start_cos_ref == pytest.approx(0.98)
        assert r.end_cos_ref == pytest.approx(0.60)

    def test_missing_endpoints_do_not_break_the_aggregate(self):
        agg = TestAggregateMath._load_helper()
        seg = TestAggregateMath._seg
        r = agg([seg(0, 50, mean=0.8, ref=0.8)])
        assert r.start_cos_ref is None and r.end_cos_ref is None
        assert r.mean_cos == pytest.approx(0.8)
