"""Tests for the segment run-time estimator (wanly-api#166).

These pin the decisions that were made against production data rather than the arithmetic, which
is trivial. Each one guards a specific way the estimator used to be wrong: a rate keyed on a
worker name that gets deleted when a pod drains, a single observation speaking with authority, a
global average that answered "how long will 720x1056 take" with a number drawn mostly from 480p.

Pure logic — no HTTP, no database.
"""

import inspect
import math

import pytest

from app import estimation
from app.estimation import (
    DURATION_EXPONENT,
    MIN_SAMPLES,
    Estimate,
    PixelLaw,
    estimate_segment,
    estimate_segment_time,
)

SHAPE = (720, 1056, 30)


def rates(*, shape=None, gpu=None, factors=None, law=None):
    return {
        "shape_rates": shape or {},
        "gpu_rates": gpu or {},
        "gpu_factors": factors or {},
        "pixel_law": law,
    }


class TestTierPriority:
    def test_gpu_at_this_shape_beats_everything(self):
        r = rates(
            shape={SHAPE: (10.0, 40)},
            gpu={(*SHAPE, "RTX 4090"): (2.0, 5)},
            factors={"RTX 4090": 0.7},
        )
        est = estimate_segment(r, *SHAPE, 5.0, "RTX 4090")
        assert est.source == "gpu"
        assert est.seconds == pytest.approx(2.0 * 5.0**DURATION_EXPONENT, abs=0.2)

    def test_known_gpu_without_this_shape_scales_the_pooled_rate(self):
        """The point of the ratio: a 4090 that has never run this shape is still not a 3090."""
        r = rates(shape={SHAPE: (10.0, 40)}, factors={"RTX 4090": 0.7})
        est = estimate_segment(r, *SHAPE, 5.0, "RTX 4090")
        assert est.source == "gpu_scaled"
        assert est.seconds == pytest.approx(7.0 * 5.0**DURATION_EXPONENT, abs=0.2)

    def test_unknown_gpu_falls_through_to_the_pooled_rate_unscaled(self):
        r = rates(shape={SHAPE: (10.0, 40)}, factors={"RTX 4090": 0.7})
        est = estimate_segment(r, *SHAPE, 5.0, "some brand new card")
        assert est.source == "shape"
        assert est.seconds == pytest.approx(10.0 * 5.0**DURATION_EXPONENT, abs=0.2)

    def test_unseen_shape_extrapolates_along_the_pixel_law(self):
        """No history for this shape, but pixels predict rate. Better than nothing and better
        than a global average, and it says so in `source`."""
        # rate = e^intercept * pixels^slope, chosen so 720*1056 pixels gives exactly 1.0
        law = PixelLaw(1.0, -math.log(720 * 1056), 12, 100_000, 2_000_000)
        est = estimate_segment(rates(law=law), *SHAPE, 5.0, None)
        assert est.source == "pixel_law"
        assert est.seconds == pytest.approx(5.0**DURATION_EXPONENT, abs=0.2)

    def test_nothing_at_all_returns_none(self):
        assert estimate_segment(rates(), *SHAPE, 5.0, None) is None
        assert estimate_segment_time(rates(), *SHAPE, 5.0, None) is None


class TestMinimumSamples:
    def test_a_shape_seen_once_does_not_speak_for_itself(self):
        """One run might have been a stall. Below the threshold it is not evidence."""
        assert estimate_segment(rates(shape={SHAPE: (10.0, 1)}), *SHAPE, 5.0, None) is None

    def test_thin_gpu_data_falls_through_rather_than_overriding(self):
        r = rates(shape={SHAPE: (10.0, 40)}, gpu={(*SHAPE, "RTX 4090"): (2.0, 1)})
        est = estimate_segment(r, *SHAPE, 5.0, "RTX 4090")
        assert est.source == "shape"

    def test_sample_count_is_reported(self):
        """#287's stats endpoint returns `samples` so the caller can judge; so does this."""
        est = estimate_segment(rates(shape={SHAPE: (10.0, 40)}), *SHAPE, 5.0, None)
        assert est.samples == 40


class TestNoGlobalAverage:
    def test_there_is_no_ungrouped_fallback_rate(self):
        """704x480 runs 329s and 720x1056 runs 1774s. One number covering both is not a weak
        estimate, it is a wrong one, and it was served exactly when a shape was new."""
        source = inspect.getsource(estimation)
        assert "global_rate" not in source

    def test_a_lone_unrelated_shape_cannot_price_a_new_one(self):
        """Without enough shapes to fit a slope, the law does not fire and the answer is None."""
        r = rates(shape={(704, 480, 60): (10.0, 40)})
        assert estimate_segment(r, *SHAPE, 5.0, None) is None


class TestDurationModel:
    def test_run_time_grows_faster_than_duration(self):
        """Measured at 1056x720: 812s of sampling at 3s against 1666s at 5s, a 2.05x cost for a
        1.67x duration. A per-second rate would price the 5s segment 18% low."""
        r = rates(shape={SHAPE: (100.0, 40)})
        three = estimate_segment_time(r, *SHAPE, 3.0, None)
        five = estimate_segment_time(r, *SHAPE, 5.0, None)
        assert five / three > 5.0 / 3.0

    def test_there_is_no_fixed_overhead_term(self):
        """Fitting a shared intercept across the eight shapes with more than one duration puts it
        at -326s, so a fixed post-sampling tail is not just unsupported, the sign is wrong."""
        r = rates(shape={SHAPE: (100.0, 40)})
        assert estimate_segment_time(r, *SHAPE, 0.5, None) < estimate_segment_time(
            r, *SHAPE, 1.0, None
        )
        # Doubling the duration from zero-ish must not converge on a constant.
        assert estimate_segment_time(r, *SHAPE, 0.001, None) == pytest.approx(0.0, abs=0.05)

    @pytest.mark.parametrize("duration", [0, 0.0, None, -1.0])
    def test_impossible_durations_are_not_priced(self, duration):
        assert estimate_segment(rates(shape={SHAPE: (10.0, 40)}), *SHAPE, duration, None) is None


class TestGpuFactors:
    def test_factor_is_the_median_ratio_against_the_pooled_rate(self):
        shape = {(720, 1056, 30): (100.0, 40), (1056, 720, 30): (200.0, 40)}
        gpu = {
            (720, 1056, 30, "RTX 4090"): (70.0, MIN_SAMPLES),
            (1056, 720, 30, "RTX 4090"): (150.0, MIN_SAMPLES),
        }
        factors = estimation._gpu_factors(shape, gpu)
        assert factors["RTX 4090"] == pytest.approx((0.7 + 0.75) / 2)

    def test_thin_gpu_groups_do_not_contribute_to_the_ratio(self):
        shape = {(720, 1056, 30): (100.0, 40)}
        gpu = {(720, 1056, 30, "RTX 4090"): (70.0, 1)}
        assert estimation._gpu_factors(shape, gpu) == {}


class TestPixelLaw:
    def test_recovers_a_known_power_law(self):
        """rate = 2e-4 * pixels ** 1.05, sampled at five shapes."""
        shape_rates = {
            (w, h, 60): (2e-4 * (w * h) ** 1.05, 10)
            for w, h in [(480, 640), (704, 480), (720, 1056), (1216, 832), (960, 1280)]
        }
        law = estimation._fit_pixel_law(shape_rates)
        assert law.slope == pytest.approx(1.05, abs=0.01)
        assert math.exp(law.intercept) == pytest.approx(2e-4, rel=0.05)
        assert law.shapes == 5
        assert (law.min_pixels, law.max_pixels) == (480 * 640, 960 * 1280)

    def test_extrapolation_is_clamped_to_the_measured_pixel_range(self):
        """The slope is only stable-ish - 0.71 to 1.51 across rolling 30 day windows - so a
        shape far outside the fitted band is priced as the largest one that was fitted."""
        law = PixelLaw(
            slope=1.6, intercept=-17.0, shapes=20, min_pixels=307_200, max_pixels=921_600
        )
        assert law.rate(2_000_000) == pytest.approx(law.rate(921_600))
        assert law.rate(100_000) == pytest.approx(law.rate(307_200))
        assert law.rate(600_000) > law.rate(307_200)

    def test_too_few_shapes_is_no_law(self):
        shape_rates = {(480, 640, 60): (10.0, 10), (704, 480, 60): (12.0, 10)}
        assert estimation._fit_pixel_law(shape_rates) is None

    def test_shapes_below_the_sample_floor_do_not_count_toward_the_fit(self):
        shape_rates = {
            (w, h, 60): (2e-4 * (w * h) ** 1.05, 1)
            for w, h in [(480, 640), (704, 480), (720, 1056), (1216, 832), (960, 1280)]
        }
        assert estimation._fit_pixel_law(shape_rates) is None


class TestWorkerNameIsGone:
    def test_the_estimator_never_keys_on_worker_name(self):
        """Worker rows are deleted when a RunPod pod drains and the names are never reused, so a
        rate keyed on one is discarded the moment the pod that earned it goes away."""
        assert "Segment.worker_name" not in inspect.getsource(estimation)


class TestEstimateShape:
    def test_estimate_is_immutable(self):
        est = Estimate(1.0, "shape", 3)
        with pytest.raises(Exception):
            est.seconds = 2.0
