"""Pricing a segment: how long is this one going to take to run?

The number feeds the queue ETA in the console, so the failure that matters is not imprecision,
it is confidence about a shape or a GPU the estimate was never drawn from. Everything here was
fitted against the 3280 completed segments in production rather than reasoned about, and the
measurements are recorded next to the constants they justify so a future change has something to
argue with.

Shape of the model. Run time is a per-shape rate multiplied by the segment's duration raised to
DURATION_EXPONENT. The obvious alternative - a fixed per-segment overhead plus a per-second
sampling rate, on the theory that decode, RIFE, stitch, faceswap and identity scoring cost the
same whatever the length - does not survive contact with the data, twice over. Fitting a shared
intercept across the eight shapes that have been run at more than one duration puts it at -326s
(bootstrap 95% CI -423 to -217), so the sign is wrong, and it buys an R2 of 0.8864 against 0.8820
for the zero-intercept fit. Parsing the phase timestamps out of progress_log on 2341 segments
says why: the post-sampling tail is a median 41s, only 9.2% of run time, and it is not fixed -
at 1056x720 it runs 28s / 59s / 81s for 1.5s / 3s / 5s segments, because the work in it is
per-frame too. There is a genuinely fixed cost, the setup before ComfyUI is handed the job, and
it is a median 3 seconds. Nothing worth modelling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from uuid import UUID

from sqlalchemy import Float, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.enums import SegmentStatus
from app.models import Job, Segment

# Linear in duration. NOT fitted — this is a neutral prior, and saying so is the point.
#
# It was 1.4, fitted on WAN 2.2: run time grew faster than duration because attention over the
# video tokens did, measured at 1056x720 on 3s (n=41) against 5s (n=102), 812s against 1666s.
# That number described WAN, and it is not transferable. Two reasons:
#
#   1. Every LTX render is a fixed 241 frames. Duration barely varies, so a superlinear term
#      has almost nothing to act on and the exponent is being asked a question the data does
#      not pose.
#   2. The WAN measurement was taken on a two-pass sampler with a high/low model split. LTX
#      samples differently. Carrying the constant across would be assuming the shape of a curve
#      from a pipeline that no longer runs.
#
# 1.0 makes the rate plain seconds-per-second, which is the honest position while there is no
# LTX history to fit against: wrong in a way that is obvious and neutral, rather than wrong in a
# way that looks measured. Re-fit it once LTX segments have accumulated across more than one
# duration — and if they never do, because 241 frames stays fixed, then the right answer is to
# drop the duration term entirely rather than to pick a number for it.
DURATION_EXPONENT = 1.0

# Only recent runs. Model, sampler and step count all drift, and a rate learned in February is
# describing a different pipeline. Backtesting says this is the single largest win available
# here: 30 days over all history moves median absolute error from 16.9% to 14.3% and p90 from
# 51.5% to 43.1%. 60 and 90 day windows both come out worse (14.5%/49.4% and 15.7%/52.6%).
WINDOW_DAYS = 30

# How many runs before a group is allowed to speak for itself. One observation produces a
# confident-looking rate from what might be a stall. Three is the measured sweet spot - at one
# sample median error is 13.9% but p90 is 45.4%, at five it is 14.7%/41.6%, and three sits at
# 14.3%/43.1% while sending fewer shapes down to the fallback.
MIN_SAMPLES = 3

# The pixel law needs enough shapes to have a slope worth trusting rather than two points and a
# ruler.
MIN_PIXEL_LAW_SHAPES = 5


@dataclass(frozen=True)
class PixelLaw:
    """rate = exp(intercept) * pixels ** slope, plus the pixel range it was fitted over."""

    slope: float
    intercept: float
    shapes: int
    min_pixels: int
    max_pixels: int

    def rate(self, pixels: int) -> float:
        # Clamped to the fitted range. The slope is only stable-ish: across rolling 30 day
        # windows of the corpus it wanders between 0.71 and 1.51 (median 1.05, and 1.67 right
        # now, because the last month has been a narrow band of 307k to 922k pixel shapes).
        # Interpolating inside the measured band with a slope like that is fine; extrapolating a
        # long way outside it compounds the error, so a shape twice as large as anything ever
        # run gets priced as the largest thing ever run. That reads low, which is the failure
        # this module keeps choosing.
        bounded = min(max(pixels, self.min_pixels), self.max_pixels)
        return math.exp(self.intercept + self.slope * math.log(bounded))


@dataclass(frozen=True)
class Estimate:
    """A priced segment and where the price came from.

    `source` and `samples` exist so a caller can tell "measured on this GPU at this shape, 40
    times" from "extrapolated from other shapes because we have never run this one", which are
    the same float and very different claims.
    """

    seconds: float
    source: str
    samples: int


async def get_estimation_rates(db: AsyncSession, user_id: UUID) -> dict:
    """Fit the estimator against this user's recent completed segments.

    Returns a dict with:
      - shape_rates: {(width, height, fps): (rate, samples)} pooled across GPUs
      - gpu_rates: {(width, height, fps, gpu_name): (rate, samples)}
      - gpu_factors: {gpu_name: multiplier against the pooled rate}
      - pixel_law: PixelLaw | None - rate as a function of the shape's pixel count
    """
    elapsed = func.extract("epoch", Segment.completed_at - Segment.claimed_at).cast(Float)
    rate = elapsed / func.power(Segment.duration_seconds, DURATION_EXPONENT)
    cutoff = datetime.now(timezone.utc) - timedelta(days=WINDOW_DAYS)

    base = (
        select()
        .select_from(Segment)
        .join(Job, Segment.job_id == Job.id)
        .where(
            Job.user_id == user_id,
            Segment.status == SegmentStatus.COMPLETED,
            Segment.claimed_at.isnot(None),
            Segment.completed_at.isnot(None),
            Segment.completed_at >= cutoff,
            Segment.duration_seconds > 0,
        )
    )

    # Median, not mean, matching the stats endpoint (wanly-console#287) so the two views cannot
    # disagree about the same segments. Honesty about what it buys: on median absolute error the
    # mean is actually the better predictor (12.9% against 14.3%), because the run-time
    # distribution is right skewed and the mean leans into that. The median wins where it was
    # meant to, in the tail - p90 43.1% against 44.5% - and one stalled run cannot move it at
    # all, which is the property being paid for.
    median_rate = func.percentile_cont(0.5).within_group(rate)

    shape_result = await db.execute(
        base.add_columns(Job.width, Job.height, Job.fps, median_rate, func.count()).group_by(
            Job.width, Job.height, Job.fps
        )
    )
    shape_rates: dict[tuple, tuple[float, int]] = {}
    for width, height, fps, value, samples in shape_result.all():
        if value is not None and value > 0:
            shape_rates[(width, height, fps)] = (float(value), samples)

    # Keyed on gpu_name, not worker_name. A RunPod worker's row is deleted when the pod drains
    # and its name is never reused, so every rate learned from a pod used to die with it; the GPU
    # is what determines speed and is snapshotted onto the segment. Note this tier can only fire
    # for a segment already claimed - a pending one has no GPU yet - which is correct: the right
    # price for work nobody has picked up is the pooled rate across whatever might pick it up.
    gpu_result = await db.execute(
        base.add_columns(
            Job.width, Job.height, Job.fps, Segment.gpu_name, median_rate, func.count()
        )
        .where(Segment.gpu_name.isnot(None))
        .group_by(Job.width, Job.height, Job.fps, Segment.gpu_name)
    )
    gpu_rates: dict[tuple, tuple[float, int]] = {}
    for width, height, fps, gpu_name, value, samples in gpu_result.all():
        if value is not None and value > 0:
            gpu_rates[(width, height, fps, gpu_name)] = (float(value), samples)

    return {
        "shape_rates": shape_rates,
        "gpu_rates": gpu_rates,
        "gpu_factors": _gpu_factors(shape_rates, gpu_rates),
        "pixel_law": _fit_pixel_law(shape_rates),
    }


def _gpu_factors(shape_rates: dict, gpu_rates: dict) -> dict[str, float]:
    """How much faster or slower each GPU is than the pool, measured rather than guessed.

    This is what lets a GPU with no history at a shape still be priced better than the pool
    average: take the shape's pooled rate and scale it. The ticket estimated the 3090 to 4090 gap
    at roughly 2x from a single pair of runs; across the shapes where both have been measured it
    is 0.658 at 1056x720 and 0.777 at 960x720, so nearer 1.4x. That is the sort of correction
    that only shows up once the ratio is computed instead of eyeballed.
    """
    ratios: dict[str, list[float]] = {}
    for (width, height, fps, gpu_name), (gpu_rate, samples) in gpu_rates.items():
        if samples < MIN_SAMPLES:
            continue
        pooled = shape_rates.get((width, height, fps))
        if pooled and pooled[1] >= MIN_SAMPLES and pooled[0] > 0:
            ratios.setdefault(gpu_name, []).append(gpu_rate / pooled[0])
    return {gpu_name: _median(values) for gpu_name, values in ratios.items() if values}


def _fit_pixel_law(shape_rates: dict) -> PixelLaw | None:
    """Fit log(rate) = intercept + slope * log(pixels) across the shapes that have been run.

    This replaces the old global average as the last resort, and it is not a small difference.
    On exactly the segments that reach the fallback - a shape with no history of its own, which
    is when the user has least of their own information - the pixel law lands at 22.3% median
    absolute error against 38.7% for a flat global rate, and 53.1% at p90 against 75.8%. Fitted
    over the whole corpus the slope is 1.02, so run time is close to linear in pixel count, which
    is the reassuring part: the fallback is following a real relationship rather than smearing
    480p and 720p together and calling the result an estimate.

    Those figures are WAN 2.2's, and they are kept because they are the argument for the SHAPE
    of this fallback - a pixel law rather than a flat average - which is not engine-specific.
    The slope and intercept are not: they are re-fitted on every call from whatever shapes are
    in the window, so once the window holds LTX segments the fallback describes LTX. Nothing
    here needs changing for that to happen; it needed only that the WAN rates stop being in the
    window, which is what clearing the history does.

    Until at least MIN_PIXEL_LAW_SHAPES LTX shapes have MIN_SAMPLES runs each, this returns
    None and callers get no estimate. That is the correct answer to "how long will this take"
    when nothing comparable has been run yet.
    """
    points = [
        (math.log(width * height), math.log(value), samples)
        for (width, height, _fps), (value, samples) in shape_rates.items()
        if samples >= MIN_SAMPLES and width and height and value > 0
    ]
    if len(points) < MIN_PIXEL_LAW_SHAPES:
        return None

    # Weighted by sample count so a shape run 650 times outvotes one run four times.
    weight = sum(n for _, _, n in points)
    mean_x = sum(x * n for x, _, n in points) / weight
    mean_y = sum(y * n for _, y, n in points) / weight
    variance = sum(n * (x - mean_x) ** 2 for x, _, n in points)
    if variance <= 0:  # every shape has the same pixel count, so there is no slope to find
        return None
    slope = sum(n * (x - mean_x) * (y - mean_y) for x, y, n in points) / variance
    return PixelLaw(
        slope=slope,
        intercept=mean_y - slope * mean_x,
        shapes=len(points),
        min_pixels=round(math.exp(min(x for x, _, _ in points))),
        max_pixels=round(math.exp(max(x for x, _, _ in points))),
    )


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def estimate_segment(
    rates: dict,
    width: int,
    height: int,
    fps: int,
    duration_seconds: float,
    gpu_name: str | None = None,
) -> Estimate | None:
    """Price one segment, most specific evidence first.

    GPU at this shape, then the shape pooled across GPUs (scaled by how fast this GPU runs, when
    that is known), then extrapolated along the pixel law, then nothing. There is deliberately no
    global-average tier: a single number covering 704x480 at 329s and 720x1056 at 1774s is not a
    weak estimate, it is a wrong one, and it was being served precisely when the shape was new.
    """
    if not duration_seconds or duration_seconds <= 0:
        return None

    scaled_duration = duration_seconds**DURATION_EXPONENT

    if gpu_name:
        measured = rates["gpu_rates"].get((width, height, fps, gpu_name))
        if measured and measured[1] >= MIN_SAMPLES:
            return Estimate(round(measured[0] * scaled_duration, 1), "gpu", measured[1])

    pooled = rates["shape_rates"].get((width, height, fps))
    if pooled and pooled[1] >= MIN_SAMPLES:
        factor = rates["gpu_factors"].get(gpu_name) if gpu_name else None
        if factor:
            seconds = pooled[0] * factor * scaled_duration
            return Estimate(round(seconds, 1), "gpu_scaled", pooled[1])
        return Estimate(round(pooled[0] * scaled_duration, 1), "shape", pooled[1])

    law = rates["pixel_law"]
    if law and width and height:
        # samples is the number of shapes the law was fitted from, not runs at this shape, of
        # which there are none. Anything reading it should be reading `source` too.
        seconds = law.rate(width * height) * scaled_duration
        return Estimate(round(seconds, 1), "pixel_law", law.shapes)

    return None


def estimate_segment_time(
    rates: dict,
    width: int,
    height: int,
    fps: int,
    duration_seconds: float,
    gpu_name: str | None = None,
) -> float | None:
    """Estimated run time in seconds, or None when there is nothing to base one on."""
    estimate = estimate_segment(rates, width, height, fps, duration_seconds, gpu_name)
    return estimate.seconds if estimate else None


def estimate_queue_drain_seconds(
    rates: dict, segments, worker_count: int, now: datetime
) -> float:
    """Wall-clock seconds until the queue is empty, not the sum of the work in it.

    Each row is (width, height, fps, duration_seconds, gpu_name, status, claimed_at).

    Two corrections to a straight sum, both of which made the old number wrong in the
    direction that matters:

    WORKERS. The sum is the work; the wait is the work divided by the machines doing it. With
    one worker the two agree, which is why this was never noticed — and then a pod is launched
    and the real wait halves while the number does not move. Divided by the workers actually
    able to claim, floored at one so an empty fleet reports the work rather than infinity.

    WORK ALREADY DONE. A segment being rendered was counted at its full estimate however long
    it had been running. Measured live: 235s into a 586s render, counted as 586. That is 1% of
    a long queue and all of a short one -- the last segment at 90% reported ten minutes with
    one to go, so the number stopped converging on zero exactly when someone was watching it
    to decide whether to queue more.

    A segment the estimator cannot price still contributes nothing rather than a guess. That
    makes the total read low on a cold database instead of confidently wrong, which is the
    safer failure for a number used to decide whether there is time for more work.
    """
    total = 0.0
    longest = 0.0
    for width, height, fps, duration_seconds, gpu_name, status, claimed_at in segments:
        if not duration_seconds:
            continue
        est = estimate_segment_time(rates, width, height, fps, duration_seconds, gpu_name)
        if not est:
            continue
        if claimed_at is not None:
            # Assume UTC for a naive timestamp rather than raising. The column is
            # `timestamp with time zone` and the driver returns aware datetimes, so this
            # should not happen -- but "should not happen" here means TypeError inside
            # GET /stats, which takes the whole dashboard down for a number that is
            # decoration. Every timestamp this system writes is UTC, so the assumption is
            # the correct one and not a guess.
            if claimed_at.tzinfo is None:
                claimed_at = claimed_at.replace(tzinfo=timezone.utc)
            # Never negative: a segment running longer than its estimate is late, not
            # ahead, and a negative would silently pay for some other segment's time.
            est = max(0.0, est - (now - claimed_at).total_seconds())
        total += est
        longest = max(longest, est)

    # A segment cannot be split across workers. With fewer segments than workers, dividing
    # says half a render time for one render — so the wait is never shorter than the longest
    # single segment left. That is the case that matters right after a pod joins an
    # almost-empty queue, which is exactly when someone is watching this number.
    return round(max(total / max(1, worker_count), longest), 1)
