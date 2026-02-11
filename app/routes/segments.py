from datetime import datetime, timezone
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.auth import get_current_user
from app.database import get_db
from app.models import Job, Segment, User, Video
from app.schemas.segments import SegmentClaimResponse, SegmentCreate, SegmentResponse, SegmentStatusUpdate

router = APIRouter()


@router.post("/jobs/{job_id}/segments", response_model=SegmentResponse, status_code=status.HTTP_201_CREATED)
async def add_segment(
    job_id: UUID,
    body: SegmentCreate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Job)
        .where(Job.id == job_id, Job.user_id == user.id)
        .options(selectinload(Job.segments))
    )
    job = result.scalar_one_or_none()
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    if job.status != "awaiting":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Job must be in 'awaiting' status to add segments (current: '{job.status}')",
        )

    next_index = max((s.index for s in job.segments), default=-1) + 1

    segment = Segment(
        job_id=job.id,
        index=next_index,
        prompt=body.prompt,
        duration_seconds=body.duration_seconds,
        start_image=body.start_image,
        loras=body.loras,
        faceswap_enabled=body.faceswap_enabled,
        faceswap_method=body.faceswap_method,
        faceswap_source_type=body.faceswap_source_type,
        faceswap_image=body.faceswap_image,
        faceswap_faces_order=body.faceswap_faces_order,
        faceswap_faces_index=body.faceswap_faces_index,
        auto_finalize=body.auto_finalize,
    )
    db.add(segment)

    job.status = "processing"

    await db.commit()
    await db.refresh(segment)
    return segment


@router.get("/segments/next")
async def claim_next_segment(
    worker_id: UUID = Query(...),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Segment)
        .join(Job, Segment.job_id == Job.id)
        .where(Segment.status == "pending", Job.status.in_(["pending", "processing"]))
        .order_by(Segment.created_at)
        .limit(1)
        .with_for_update(skip_locked=True)
    )
    segment = result.scalar_one_or_none()
    if segment is None:
        return None

    now = datetime.now(timezone.utc)
    segment.status = "claimed"
    segment.worker_id = worker_id
    segment.claimed_at = now

    job = await db.get(Job, segment.job_id)
    if job.status == "pending":
        job.status = "processing"

    # Resolve start_image
    resolved_start_image = segment.start_image
    if resolved_start_image is None:
        if segment.index == 0:
            resolved_start_image = job.starting_image
        else:
            prev_result = await db.execute(
                select(Segment)
                .where(Segment.job_id == job.id, Segment.index == segment.index - 1)
            )
            prev_segment = prev_result.scalar_one_or_none()
            if prev_segment is not None:
                resolved_start_image = prev_segment.last_frame_path

    await db.commit()
    await db.refresh(segment)

    return SegmentClaimResponse(
        id=segment.id,
        job_id=segment.job_id,
        index=segment.index,
        prompt=segment.prompt,
        duration_seconds=segment.duration_seconds,
        start_image=resolved_start_image,
        loras=segment.loras,
        faceswap_enabled=segment.faceswap_enabled,
        faceswap_method=segment.faceswap_method,
        faceswap_source_type=segment.faceswap_source_type,
        faceswap_image=segment.faceswap_image,
        faceswap_faces_order=segment.faceswap_faces_order,
        faceswap_faces_index=segment.faceswap_faces_index,
        width=job.width,
        height=job.height,
        fps=job.fps,
        seed=job.seed,
    )


@router.patch("/segments/{segment_id}", response_model=SegmentResponse)
async def update_segment(
    segment_id: UUID,
    body: SegmentStatusUpdate,
    db: AsyncSession = Depends(get_db),
):
    segment = await db.get(Segment, segment_id)
    if segment is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Segment not found")

    if body.status is not None:
        segment.status = body.status
        if body.status in ("completed", "failed"):
            segment.completed_at = datetime.now(timezone.utc)
    if body.output_path is not None:
        segment.output_path = body.output_path
    if body.last_frame_path is not None:
        segment.last_frame_path = body.last_frame_path
    if body.error_message is not None:
        segment.error_message = body.error_message

    await db.flush()

    # Check if job needs status update
    if body.status in ("completed", "failed"):
        job = await db.get(Job, segment.job_id)
        result = await db.execute(
            select(Segment).where(
                Segment.job_id == job.id,
                Segment.status.in_(["pending", "claimed", "processing"]),
            )
        )
        active_segments = result.scalars().all()
        if len(active_segments) == 0:
            if segment.auto_finalize and body.status == "completed":
                job.status = "finalized"
                video = Video(job_id=job.id, status="pending")
                db.add(video)
            else:
                job.status = "awaiting"

    await db.commit()
    await db.refresh(segment)
    return segment
