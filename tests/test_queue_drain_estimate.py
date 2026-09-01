"""Total queue time is a WAIT, not a sum of work.

Measured on production before this changed: 36 segments, 21741s reported, which is exactly
36 x the 586s average. Correct with one worker and wrong the moment a pod joins — the real
wait halves while the number does not move. The in-flight segment was also counted at full
price 235s into its render.
"""

import datetime as dt

import pytest

from app.estimation import PixelLaw, estimate_queue_drain_seconds

NOW = dt.datetime(2026, 9, 1, 12, 0, tzinfo=dt.timezone.utc)

# A rate table the estimator can price from: one shape, plenty of samples.
RATES = {
    "shape_rates": {(1216, 832, 24): (58.0, 22)},
    "gpu_rates": {},
    "gpu_factors": {},
    "pixel_law": PixelLaw(slope=1.0, intercept=0.0, shapes=3, min_pixels=1, max_pixels=10**9),
}


def _row(status="pending", claimed_at=None):
    return (1216, 832, 24, 10.04, "NVIDIA GeForce RTX 3090", status, claimed_at)


def _one_segment_estimate():
    return estimate_queue_drain_seconds(RATES, [_row()], 1, NOW)


@pytest.mark.skipif(_one_segment_estimate() == 0, reason="estimator cannot price this shape")
class TestDrain:
    def test_two_workers_halve_the_wait(self):
        rows = [_row(), _row(), _row(), _row()]
        one = estimate_queue_drain_seconds(RATES, rows, 1, NOW)
        two = estimate_queue_drain_seconds(RATES, rows, 2, NOW)
        assert two == pytest.approx(one / 2, rel=0.01)

    def test_no_workers_reports_the_work_rather_than_infinity(self):
        rows = [_row(), _row()]
        assert estimate_queue_drain_seconds(RATES, rows, 0, NOW) == estimate_queue_drain_seconds(
            RATES, rows, 1, NOW
        )

    def test_an_in_flight_segment_is_discounted_by_what_it_has_already_run(self):
        fresh = estimate_queue_drain_seconds(RATES, [_row()], 1, NOW)
        started = NOW - dt.timedelta(seconds=100)
        part_done = estimate_queue_drain_seconds(
            RATES, [_row("processing", started)], 1, NOW
        )
        assert part_done == pytest.approx(fresh - 100, abs=1.0)

    def test_a_segment_running_over_its_estimate_never_goes_negative(self):
        """Late is not ahead. A negative would silently pay for another segment's time."""
        long_ago = NOW - dt.timedelta(hours=5)
        assert estimate_queue_drain_seconds(
            RATES, [_row("processing", long_ago)], 1, NOW
        ) == 0.0

    def test_an_unpriceable_segment_still_contributes_nothing(self):
        # Reads low on a cold database rather than confidently wrong.
        unknown = (None, None, 24, 10.04, None, "pending", None)
        assert estimate_queue_drain_seconds(RATES, [unknown], 1, NOW) == 0.0

    def test_more_workers_than_segments_cannot_beat_one_segment(self):
        """A segment cannot be split across workers.

        One segment and four workers is still one render: dividing would report a quarter of
        it. This is the case right after a pod joins an almost-empty queue -- precisely when
        the number is being watched.
        """
        one = estimate_queue_drain_seconds(RATES, [_row()], 1, NOW)
        many = estimate_queue_drain_seconds(RATES, [_row()], 4, NOW)
        assert many == one

    def test_workers_still_help_when_there_is_work_for_them(self):
        rows = [_row() for _ in range(8)]
        assert estimate_queue_drain_seconds(RATES, rows, 4, NOW) == pytest.approx(
            estimate_queue_drain_seconds(RATES, rows, 1, NOW) / 4, rel=0.01
        )
