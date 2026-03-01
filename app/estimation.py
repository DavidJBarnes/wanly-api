from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Job, Segment


async def get_estimation_rates(
    db: AsyncSession, user_id: UUID
) -> dict:
    """Compute average run-time-per-second rates from completed segments.

    Returns dict with:
      - rates: {(width, height, fps): rate} — config-level
      - worker_rates: {(width, height, fps, worker_name): rate} — worker+config
      - global_rate: float | None — ungrouped fallback
    """
    run_time_expr = (
        func.extract("epoch", Segment.completed_at)
        - func.extract("epoch", Segment.claimed_at)
    )
    base = (
        select()
        .select_from(Segment)
        .join(Job, Segment.job_id == Job.id)
        .where(
            Job.user_id == user_id,
            Segment.status == "completed",
            Segment.claimed_at.isnot(None),
            Segment.completed_at.isnot(None),
            Segment.duration_seconds > 0,
        )
    )

    # 1. Config-level rates
    config_result = await db.execute(
        base.add_columns(
            Job.width,
            Job.height,
            Job.fps,
            func.avg(run_time_expr / Segment.duration_seconds),
        ).group_by(Job.width, Job.height, Job.fps)
    )
    rates = {}
    for row in config_result.all():
        w, h, fps, avg_rate = row
        if avg_rate is not None:
            rates[(w, h, fps)] = float(avg_rate)

    # 2. Worker-level rates
    worker_result = await db.execute(
        base.add_columns(
            Job.width,
            Job.height,
            Job.fps,
            Segment.worker_name,
            func.avg(run_time_expr / Segment.duration_seconds),
        )
        .where(Segment.worker_name.isnot(None))
        .group_by(Job.width, Job.height, Job.fps, Segment.worker_name)
    )
    worker_rates = {}
    for row in worker_result.all():
        w, h, fps, worker, avg_rate = row
        if avg_rate is not None:
            worker_rates[(w, h, fps, worker)] = float(avg_rate)

    # 3. Global fallback
    global_result = await db.execute(
        base.add_columns(func.avg(run_time_expr / Segment.duration_seconds))
    )
    global_rate_val = global_result.scalar_one_or_none()
    global_rate = float(global_rate_val) if global_rate_val is not None else None

    return {
        "rates": rates,
        "worker_rates": worker_rates,
        "global_rate": global_rate,
    }


def estimate_segment_time(
    rates: dict,
    width: int,
    height: int,
    fps: int,
    duration_seconds: float,
    worker_name: str | None = None,
) -> float | None:
    """Estimate run time in seconds using best available rate.

    Priority: worker+config → config → global → None
    """
    rate = None

    if worker_name:
        rate = rates["worker_rates"].get((width, height, fps, worker_name))

    if rate is None:
        rate = rates["rates"].get((width, height, fps))

    if rate is None:
        rate = rates["global_rate"]

    if rate is None:
        return None

    return round(rate * duration_seconds, 1)
