"""Tests for the dashboard's Total Queue Time sum.

The number answers "is there room to queue more work", so the failure that matters is a
confident overestimate. An unpriceable segment therefore contributes nothing rather than a
fallback guess, and the total reads low on a cold database instead of wrong.

Pure logic — no HTTP, no database, matching the house style.
"""

import pytest

from app.estimation import sum_estimated_queue_time

# (width, height, fps, duration_seconds, worker_name)
ROW = (720, 1056, 30, 4.0, "3090.zero")


def rates(*, config=None, worker=None, global_rate=None):
    return {
        "rates": config or {},
        "worker_rates": worker or {},
        "global_rate": global_rate,
    }


class TestSumming:
    def test_empty_queue_is_zero(self):
        assert sum_estimated_queue_time(rates(global_rate=2.0), []) == 0.0

    def test_sums_across_segments(self):
        r = rates(global_rate=2.0)
        assert sum_estimated_queue_time(r, [ROW, ROW, ROW]) == pytest.approx(24.0)

    def test_mixed_configs_each_use_their_own_rate(self):
        """A 720p and a 1080p segment must not be priced with the same rate."""
        r = rates(config={(720, 1056, 30): 2.0, (1056, 720, 30): 5.0})
        total = sum_estimated_queue_time(r, [
            (720, 1056, 30, 4.0, None),
            (1056, 720, 30, 4.0, None),
        ])
        assert total == pytest.approx(8.0 + 20.0)


class TestUnpriceableSegments:
    def test_no_rate_at_all_contributes_nothing(self):
        """Cold database: better to read 0 than to invent a duration."""
        assert sum_estimated_queue_time(rates(), [ROW, ROW]) == 0.0

    def test_priceable_segments_still_count_when_others_cannot_be_priced(self):
        """One unknown config must not zero out the whole total."""
        r = rates(config={(720, 1056, 30): 2.0})
        total = sum_estimated_queue_time(r, [ROW, (9999, 9999, 30, 4.0, None)])
        assert total == pytest.approx(8.0)

    @pytest.mark.parametrize("duration", [0, 0.0, None])
    def test_segments_with_no_duration_are_skipped(self, duration):
        """duration_seconds is nullable; rate * None would raise rather than degrade."""
        r = rates(global_rate=2.0)
        assert sum_estimated_queue_time(r, [(720, 1056, 30, duration, None)]) == 0.0


class TestRateSelection:
    def test_worker_rate_wins_over_config_rate(self):
        """A known worker's measured pace is the best predictor available."""
        r = rates(
            config={(720, 1056, 30): 10.0},
            worker={(720, 1056, 30, "3090.zero"): 2.0},
        )
        assert sum_estimated_queue_time(r, [ROW]) == pytest.approx(8.0)

    def test_falls_back_to_global_when_config_is_unseen(self):
        assert sum_estimated_queue_time(rates(global_rate=3.0), [ROW]) == pytest.approx(12.0)

    def test_unclaimed_segments_use_the_config_rate(self):
        """Pending segments have no worker_name yet — the common case for a deep queue."""
        r = rates(config={(720, 1056, 30): 2.0})
        assert sum_estimated_queue_time(r, [(720, 1056, 30, 4.0, None)]) == pytest.approx(8.0)
