"""Aggregate run-time statistics.

Grouped by GPU **and** job shape, never by GPU alone. Measured on one machine in a single day:

    480x720   / 3s   ->   329s
    704x480   / 5s   ->   566s, 575s
    720x1056  / 5s   ->  1774s, 1787s

A single "average for a 4090" mixing those is noise, and worse than no number at all because it
looks authoritative. Shape is part of the key.
"""

from fastapi import APIRouter, Depends, Query
from sqlalchemy import Float, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import get_current_user
from app.database import get_db
from app.enums import SegmentStatus
from app.models import Job, Segment
from app.schemas.stats import SegmentRuntimeGroup

router = APIRouter()


@router.get("/stats/segment-runtimes", response_model=list[SegmentRuntimeGroup],
            dependencies=[Depends(get_current_user)])
async def segment_runtimes(
    db: AsyncSession = Depends(get_db),
    min_samples: int = Query(1, ge=1, description="Hide groups with fewer runs than this"),
):
    """Completed segment run times, grouped by GPU and job shape.

    Run time is completed_at - claimed_at: the whole segment as the worker experienced it,
    including the tail after the GPU goes quiet — decode, RIFE, stitching, faceswap and identity
    scoring, which together run 47s to over 2 minutes. Sampling time alone would understate what
    a job actually costs.
    """
    elapsed = func.extract("epoch", Segment.completed_at - Segment.claimed_at).cast(Float)

    stmt = (
        select(
            Segment.gpu_name,
            Job.width,
            Job.height,
            Segment.duration_seconds,
            func.count().label("samples"),
            func.avg(elapsed).label("avg_seconds"),
            func.min(elapsed).label("min_seconds"),
            func.max(elapsed).label("max_seconds"),
            # Median resists the one pathological run that a mean would hide behind. The 720p
            # segments that ran with the model fully offloaded took the same wall-clock as the
            # ones that did not — but that is not something to rely on holding.
            func.percentile_cont(0.5)
            .within_group(elapsed)
            .label("median_seconds"),
        )
        .join(Job, Job.id == Segment.job_id)
        .where(
            Segment.status == SegmentStatus.COMPLETED,
            Segment.claimed_at.isnot(None),
            Segment.completed_at.isnot(None),
            # Guard against clock skew or a repaired row producing a negative duration, which
            # would drag an average somewhere impossible.
            Segment.completed_at > Segment.claimed_at,
        )
        .group_by(Segment.gpu_name, Job.width, Job.height, Segment.duration_seconds)
        .having(func.count() >= min_samples)
        .order_by(Segment.gpu_name, Job.width, Job.height, Segment.duration_seconds)
    )

    rows = (await db.execute(stmt)).all()
    return [
        SegmentRuntimeGroup(
            # NULL means the segment predates gpu_name, or ran on a pod whose row is gone.
            gpu_name=r.gpu_name or "unknown",
            width=r.width,
            height=r.height,
            clip_seconds=r.duration_seconds,
            samples=r.samples,
            avg_seconds=round(r.avg_seconds or 0, 1),
            median_seconds=round(r.median_seconds or 0, 1),
            min_seconds=round(r.min_seconds or 0, 1),
            max_seconds=round(r.max_seconds or 0, 1),
        )
        for r in rows
    ]
