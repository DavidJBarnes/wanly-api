"""Tests for the dashboard's Total Queue Time sum.

The number answers "is there room to queue more work", so the failure that matters is a
confident overestimate. An unpriceable segment therefore contributes nothing rather than a
fallback guess, and the total reads low on a cold database instead of wrong.

Pure logic — no HTTP, no database, matching the house style.
"""

import pytest

from app.estimation import DURATION_EXPONENT, sum_estimated_queue_time

# (width, height, fps, duration_seconds, gpu_name)
ROW = (720, 1056, 30, 4.0, "NVIDIA GeForce RTX 3090")

# Run time is rate * duration ** DURATION_EXPONENT, so the tests state the rate they want and
# let this do the arithmetic rather than hard-coding numbers that move when the exponent does.
SCALED_4S = 4.0**DURATION_EXPONENT


def rates(*, shape=None, gpu=None, factors=None, law=None):
    return {
        "shape_rates": shape or {},
        "gpu_rates": gpu or {},
        "gpu_factors": factors or {},
        "pixel_law": law,
    }


class TestSumming:
    def test_empty_queue_is_zero(self):
        assert sum_estimated_queue_time(rates(shape={(720, 1056, 30): (2.0, 10)}), []) == 0.0

    def test_sums_across_segments(self):
        r = rates(shape={(720, 1056, 30): (2.0, 10)})
        assert sum_estimated_queue_time(r, [ROW, ROW, ROW]) == pytest.approx(
            round(2.0 * SCALED_4S, 1) * 3, abs=0.2
        )

    def test_mixed_shapes_each_use_their_own_rate(self):
        """A 720p and a 1080p segment must not be priced with the same rate."""
        r = rates(shape={(720, 1056, 30): (2.0, 10), (1056, 720, 30): (5.0, 10)})
        total = sum_estimated_queue_time(
            r, [(720, 1056, 30, 4.0, None), (1056, 720, 30, 4.0, None)]
        )
        assert total == pytest.approx(7.0 * SCALED_4S, abs=0.2)


class TestUnpriceableSegments:
    def test_no_rate_at_all_contributes_nothing(self):
        """Cold database: better to read 0 than to invent a duration."""
        assert sum_estimated_queue_time(rates(), [ROW, ROW]) == 0.0

    def test_priceable_segments_still_count_when_others_cannot_be_priced(self):
        """One unknown shape must not zero out the whole total."""
        r = rates(shape={(720, 1056, 30): (2.0, 10)})
        total = sum_estimated_queue_time(r, [ROW, (9999, 9999, 30, 4.0, None)])
        assert total == pytest.approx(2.0 * SCALED_4S, abs=0.2)

    @pytest.mark.parametrize("duration", [0, 0.0, None])
    def test_segments_with_no_duration_are_skipped(self, duration):
        """duration_seconds is nullable; rate * None would raise rather than degrade."""
        r = rates(shape={(720, 1056, 30): (2.0, 10)})
        assert sum_estimated_queue_time(r, [(720, 1056, 30, duration, None)]) == 0.0


class TestRateSelection:
    def test_gpu_rate_wins_over_shape_rate(self):
        """The GPU that will run it is the best predictor available."""
        r = rates(
            shape={(720, 1056, 30): (10.0, 40)},
            gpu={(720, 1056, 30, "NVIDIA GeForce RTX 3090"): (2.0, 10)},
        )
        assert sum_estimated_queue_time(r, [ROW]) == pytest.approx(2.0 * SCALED_4S, abs=0.2)

    def test_unclaimed_segments_use_the_pooled_shape_rate(self):
        """Pending segments have no gpu_name yet - the common case for a deep queue."""
        r = rates(shape={(720, 1056, 30): (2.0, 10)})
        assert sum_estimated_queue_time(r, [(720, 1056, 30, 4.0, None)]) == pytest.approx(
            2.0 * SCALED_4S, abs=0.2
        )
