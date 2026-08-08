"""Tests for run-time stats grouped by GPU and shape (wanly-console#287).

The trap this guards against is a number that looks authoritative and means nothing. Measured on
one machine in a single day:

    704x480  / 3s  ->   329s
    704x480  / 5s  ->   566s, 575s
    720x1056 / 5s  ->  1774s, 1787s

Averaging those into "a 4090 takes ~850s" would be worse than reporting nothing, because someone
would plan against it.
"""

import inspect

from app.routes import stats


class TestGrouping:
    def test_groups_by_shape_not_just_gpu(self):
        """Shape is part of the identity. A 5x spread hides inside a GPU-only average."""
        src = inspect.getsource(stats.segment_runtimes)
        assert ".group_by(" in src
        for column in ("Segment.gpu_name", "Job.width", "Job.height", "Segment.duration_seconds"):
            assert column in src.split(".group_by(")[1].split(")")[0] + ")", (
                f"{column} must be part of the grouping key"
            )

    def test_runtime_spans_the_whole_segment_not_just_sampling(self):
        """completed_at - claimed_at. The tail after the GPU idles — decode, RIFE, stitching,
        faceswap, identity scoring — runs 47s to over 2 minutes and is part of what a job costs."""
        src = inspect.getsource(stats.segment_runtimes)
        assert "Segment.completed_at - Segment.claimed_at" in src

    def test_excludes_incomplete_and_impossible_rows(self):
        src = inspect.getsource(stats.segment_runtimes)
        assert "SegmentStatus.COMPLETED" in src
        assert "Segment.claimed_at.isnot(None)" in src
        assert "Segment.completed_at.isnot(None)" in src
        # A negative duration from clock skew would drag an average somewhere impossible.
        assert "Segment.completed_at > Segment.claimed_at" in src

    def test_reports_median_alongside_mean(self):
        """One pathological run should not be able to hide behind a mean."""
        src = inspect.getsource(stats.segment_runtimes)
        assert "percentile_cont" in src


class TestSchema:
    def test_group_carries_shape_and_sample_count(self):
        from app.schemas.stats import SegmentRuntimeGroup

        fields = SegmentRuntimeGroup.model_fields
        for required in ("gpu_name", "width", "height", "clip_seconds", "samples"):
            assert required in fields, f"{required} is needed to interpret the numbers"
        # Without samples, a group built from one run looks identical to one built from fifty.
        assert fields["samples"].annotation is int
