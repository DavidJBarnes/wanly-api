import asyncio
import logging
import random
import re
from datetime import datetime, timedelta, timezone
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.auth import get_current_user
from app.database import get_db
from app.models import Job, Lora, Segment, User, Video, Wildcard
from app.s3 import delete_object
from app.schemas.segments import SegmentClaimResponse, SegmentCreate, SegmentResponse, SegmentStatusUpdate, WorkerSegmentResponse
from app.stitch import stitch_video

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


async def _resolve_wildcards(db: AsyncSession, prompt: str) -> tuple[str, str | None]:
    """Resolve <wildcard> placeholders in a prompt.

    Returns (resolved_prompt, template_or_none).
    If no wildcards found, returns (prompt, None).
    """
    pattern = re.compile(r"<([^<>]+)>")
    matches = pattern.findall(prompt)
    if not matches:
        return prompt, None

    # Fetch all referenced wildcards in one query
    unique_names = list(set(matches))
    result = await db.execute(
        select(Wildcard).where(Wildcard.name.in_(unique_names))
    )
    wildcards_by_name = {w.name: w for w in result.scalars().all()}

    template = prompt
    resolved = prompt
    for name in unique_names:
        wc = wildcards_by_name.get(name)
        if wc and wc.options:
            # Replace all occurrences of this wildcard
            chosen = random.choice(wc.options)
            resolved = resolved.replace(f"<{name}>", chosen)

    return resolved, template


@router.get("/segments", response_model=list[WorkerSegmentResponse])
async def list_segments(
    worker_id: UUID = Query(...),
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Segment, Job.name)
        .join(Job, Segment.job_id == Job.id)
        .where(Segment.worker_id == worker_id)
        .order_by(Segment.completed_at.desc().nullslast(), Segment.created_at.desc())
        .limit(limit)
    )
    rows = result.all()
    return [
        WorkerSegmentResponse(
            id=seg.id,
            job_id=seg.job_id,
            job_name=job_name,
            index=seg.index,
            prompt=seg.prompt,
            status=seg.status,
            duration_seconds=seg.duration_seconds,
            created_at=seg.created_at,
            claimed_at=seg.claimed_at,
            completed_at=seg.completed_at,
        )
        for seg, job_name in rows
    ]


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
    if job.status not in ("awaiting", "failed"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Job must be in 'awaiting' or 'failed' status to add segments (current: '{job.status}')",
        )

    next_index = max((s.index for s in job.segments), default=-1) + 1

    resolved_loras = await _resolve_loras(db, body.loras)
    resolved_prompt, prompt_template = await _resolve_wildcards(db, body.prompt)

    segment = Segment(
        job_id=job.id,
        index=next_index,
        prompt=resolved_prompt,
        prompt_template=prompt_template,
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

    job.status = "pending"

    await db.commit()
    await db.refresh(segment)
    return segment


@router.get("/segments/next")
async def claim_next_segment(
    worker_id: UUID = Query(...),
    worker_name: str = Query(None),
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
        stale.worker_name = None
        stale.claimed_at = None
        stale.progress_log = None

    result = await db.execute(
        select(Segment)
        .join(Job, Segment.job_id == Job.id)
        .where(Segment.status == "pending", Job.status.in_(["pending", "processing"]))
        .order_by(Job.priority.asc(), Segment.created_at.asc())
        .limit(1)
        .with_for_update(skip_locked=True)
    )
    segment = result.scalar_one_or_none()
    if segment is None:
        return None

    now = datetime.now(timezone.utc)
    segment.status = "claimed"
    segment.worker_id = worker_id
    segment.worker_name = worker_name
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
        initial_reference_image=job.starting_image,
        width=job.width,
        height=job.height,
        fps=job.fps,
        seed=job.seed,
    )


@router.patch("/segments/{segment_id}", response_model=SegmentResponse)
async def update_segment(
    segment_id: UUID,
    body: SegmentStatusUpdate,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    segment = await db.get(Segment, segment_id)
    if segment is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Segment not found")

    if body.status is not None:
        segment.status = body.status
        if body.status in ("completed", "failed") and segment.completed_at is None:
            segment.completed_at = datetime.now(timezone.utc)
    if body.output_path is not None:
        segment.output_path = body.output_path
    if body.last_frame_path is not None:
        segment.last_frame_path = body.last_frame_path
    if body.error_message is not None:
        segment.error_message = body.error_message
    if body.progress_log is not None:
        segment.progress_log = body.progress_log

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
                await db.flush()
                background_tasks.add_task(stitch_video, video.id, job.id)
            elif body.status == "failed":
                job.status = "failed"
            else:
                job.status = "awaiting"

    await db.commit()
    await db.refresh(segment)
    return segment


@router.post("/segments/{segment_id}/retry", response_model=SegmentResponse)
async def retry_segment(
    segment_id: UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Segment)
        .join(Job, Segment.job_id == Job.id)
        .where(Segment.id == segment_id, Job.user_id == user.id)
    )
    segment = result.scalar_one_or_none()
    if segment is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Segment not found")
    if segment.status != "failed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Only failed segments can be retried (current: '{segment.status}')",
        )

    segment.status = "pending"
    segment.worker_id = None
    segment.worker_name = None
    segment.claimed_at = None
    segment.completed_at = None
    segment.error_message = None
    segment.progress_log = None

    job = await db.get(Job, segment.job_id)
    job.status = "pending"

    await db.commit()
    await db.refresh(segment)
    return segment


@router.delete("/segments/{segment_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_segment(
    segment_id: UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Segment)
        .join(Job, Segment.job_id == Job.id)
        .where(Segment.id == segment_id, Job.user_id == user.id)
        .options(selectinload(Segment.job))
    )
    segment = result.scalar_one_or_none()
    if segment is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Segment not found")
    if segment.status not in ("failed", "completed"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Only failed or completed segments can be deleted (current: '{segment.status}')",
        )

    # Cannot delete the only segment
    job = await db.get(Job, segment.job_id)
    all_segs_result = await db.execute(
        select(Segment).where(Segment.job_id == job.id).order_by(Segment.index)
    )
    all_segs = all_segs_result.scalars().all()
    if len(all_segs) <= 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete the only segment in a job",
        )

    # S3 cleanup
    for path in [segment.output_path, segment.last_frame_path, segment.faceswap_image]:
        if path:
            try:
                await asyncio.to_thread(delete_object, path)
            except Exception:
                logger.warning("Failed to delete S3 object: %s", path)

    await db.delete(segment)
    await db.flush()

    # Re-index remaining segments
    remaining_result = await db.execute(
        select(Segment).where(Segment.job_id == job.id).order_by(Segment.index)
    )
    for i, seg in enumerate(remaining_result.scalars().all()):
        seg.index = i

    # Update job status if needed
    has_failed = await db.execute(
        select(Segment).where(Segment.job_id == job.id, Segment.status == "failed")
    )
    if job.status == "failed" and has_failed.scalar_one_or_none() is None:
        job.status = "awaiting"

    await db.commit()
