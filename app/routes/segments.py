import logging
from datetime import datetime, timedelta, timezone
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.auth import get_current_user
from app.database import get_db
from app.models import Job, Lora, Segment, User, Video
from app.schemas.segments import SegmentClaimResponse, SegmentCreate, SegmentResponse, SegmentStatusUpdate

logger = logging.getLogger(__name__)

router = APIRouter()


async def _resolve_loras(db: AsyncSession, loras_input: list | None) -> list | None:
    """Resolve lora_id references to full file info for daemon consumption."""
    if not loras_input:
        return loras_input
    resolved = []
    for item in loras_input:
        if not isinstance(item, dict):
            resolved.append(item)
            continue
        lora_id = item.get("lora_id")
        if lora_id:
            lora = await db.get(Lora, UUID(lora_id))
            if lora is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"LoRA not found: {lora_id}",
                )
            resolved.append({
                "lora_id": str(lora.id),
                "high_file": lora.high_file,
                "high_s3_uri": lora.high_s3_uri,
                "high_weight": item.get("high_weight", lora.default_high_weight),
                "low_file": lora.low_file,
                "low_s3_uri": lora.low_s3_uri,
                "low_weight": item.get("low_weight", lora.default_low_weight),
            })
        else:
            # Backward compat: raw filename format
            resolved.append(item)
    return resolved


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

    resolved_loras = await _resolve_loras(db, body.loras)

    segment = Segment(
        job_id=job.id,
        index=next_index,
        prompt=body.prompt,
        duration_seconds=body.duration_seconds,
        start_image=body.start_image,
        loras=resolved_loras,
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
    # Reset stale segments: claimed/processing for > 30 minutes with no completion
    stale_cutoff = datetime.now(timezone.utc) - timedelta(minutes=30)
    stale_result = await db.execute(
        select(Segment).where(
            Segment.status.in_(["claimed", "processing"]),
            Segment.claimed_at < stale_cutoff,
        )
    )
    for stale in stale_result.scalars().all():
        logger.warning("Resetting stale segment %s (status=%s, claimed_at=%s)", stale.id, stale.status, stale.claimed_at)
        stale.status = "pending"
        stale.worker_id = None
        stale.claimed_at = None

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
