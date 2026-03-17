import asyncio
import base64
import json
import logging
import random
import re
import subprocess
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.auth import get_current_user, verify_api_key
from app.database import get_db
from app.enums import JobStatus, SegmentStatus, VideoStatus
from app.models import AppSetting, Job, Lora, Segment, User, Video, Wildcard
from app.s3 import delete_object, download_file, move_object, parse_s3_uri
from app.schemas.segments import (
    FramePreview,
    FramePreviewResponse,
    SegmentClaimResponse,
    SegmentCreate,
    SegmentResponse,
    SegmentStatusUpdate,
    SegmentTrimUpdate,
    WorkerSegmentResponse,
)
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


@router.get("/segments", response_model=list[WorkerSegmentResponse], dependencies=[Depends(verify_api_key)])
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
    if job.status not in (JobStatus.AWAITING, JobStatus.FAILED):
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
        speed=body.speed,
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

    job.status = JobStatus.PENDING

    await db.commit()
    await db.refresh(segment)
    return segment


@router.get("/segments/next", dependencies=[Depends(verify_api_key)])
async def claim_next_segment(
    worker_id: UUID = Query(...),
    worker_name: str = Query(None),
    db: AsyncSession = Depends(get_db),
):
    # Reset stale segments: claimed/processing for > 30 minutes with no completion
    stale_cutoff = datetime.now(timezone.utc) - timedelta(minutes=30)
    stale_result = await db.execute(
        select(Segment).where(
            Segment.status.in_([SegmentStatus.CLAIMED, SegmentStatus.PROCESSING]),
            Segment.claimed_at < stale_cutoff,
        )
    )
    for stale in stale_result.scalars().all():
        logger.warning("Resetting stale segment %s (status=%s, claimed_at=%s)", stale.id, stale.status, stale.claimed_at)
        stale.status = SegmentStatus.PENDING
        stale.worker_id = None
        stale.worker_name = None
        stale.claimed_at = None
        stale.progress_log = None

    result = await db.execute(
        select(Segment)
        .join(Job, Segment.job_id == Job.id)
        .where(Segment.status == SegmentStatus.PENDING, Job.status.in_([JobStatus.PENDING, JobStatus.PROCESSING]))
        .order_by(Job.priority.asc(), Segment.created_at.asc())
        .limit(1)
        .with_for_update(skip_locked=True)
    )
    segment = result.scalar_one_or_none()
    if segment is None:
        return None

    now = datetime.now(timezone.utc)
    segment.status = SegmentStatus.CLAIMED
    segment.worker_id = worker_id
    segment.worker_name = worker_name
    segment.claimed_at = now

    job = await db.get(Job, segment.job_id)
    if job.status == JobStatus.PENDING:
        job.status = JobStatus.PROCESSING

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

    # Fetch negative_prompt from app settings
    neg_setting = await db.get(AppSetting, "negative_prompt")
    negative_prompt = neg_setting.value if neg_setting else None

    await db.commit()
    await db.refresh(segment)

    return SegmentClaimResponse(
        id=segment.id,
        job_id=segment.job_id,
        index=segment.index,
        prompt=segment.prompt,
        duration_seconds=segment.duration_seconds,
        speed=segment.speed,
        start_image=resolved_start_image,
        loras=segment.loras,
        faceswap_enabled=segment.faceswap_enabled,
        faceswap_method=segment.faceswap_method,
        faceswap_source_type=segment.faceswap_source_type,
        faceswap_image=segment.faceswap_image,
        faceswap_faces_order=segment.faceswap_faces_order,
        faceswap_faces_index=segment.faceswap_faces_index,
        initial_reference_image=job.starting_image,
        lightx2v_strength_high=job.lightx2v_strength_high,
        lightx2v_strength_low=job.lightx2v_strength_low,
        cfg_high=job.cfg_high,
        cfg_low=job.cfg_low,
        negative_prompt=negative_prompt,
        width=job.width,
        height=job.height,
        fps=job.fps,
        seed=job.seed,
    )


@router.patch("/segments/{segment_id}", response_model=SegmentResponse, dependencies=[Depends(verify_api_key)])
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
        if body.status in (SegmentStatus.COMPLETED, SegmentStatus.FAILED) and segment.completed_at is None:
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
    if body.status in (SegmentStatus.COMPLETED, SegmentStatus.FAILED):
        job = await db.get(Job, segment.job_id)
        result = await db.execute(
            select(Segment).where(
                Segment.job_id == job.id,
                Segment.status.in_([SegmentStatus.PENDING, SegmentStatus.CLAIMED, SegmentStatus.PROCESSING]),
            )
        )
        active_segments = result.scalars().all()
        if len(active_segments) == 0:
            if segment.auto_finalize and body.status == SegmentStatus.COMPLETED:
                job.status = JobStatus.FINALIZED
                video = Video(job_id=job.id, status=VideoStatus.PENDING)
                db.add(video)
                await db.flush()
                background_tasks.add_task(stitch_video, video.id, job.id)
            elif body.status == SegmentStatus.FAILED:
                job.status = JobStatus.FAILED
            else:
                job.status = JobStatus.AWAITING

    await db.commit()
    await db.refresh(segment)
    return segment


@router.patch("/segments/{segment_id}/transition", response_model=SegmentResponse)
async def update_segment_transition(
    segment_id: UUID,
    body: dict,
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

    transition = body.get("transition")
    if transition is not None and transition not in ("fade", "flash"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid transition: {transition}")

    segment.transition = transition
    await db.commit()
    await db.refresh(segment)
    return segment


@router.patch("/segments/{segment_id}/trim", response_model=SegmentResponse)
async def update_segment_trim(
    segment_id: UUID,
    body: SegmentTrimUpdate,
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

    job = await db.get(Job, segment.job_id)
    total_frames = int(segment.duration_seconds * job.fps)
    if body.trim_start_frames + body.trim_end_frames >= total_frames:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Trim exceeds total frames ({total_frames})",
        )

    segment.trim_start_frames = body.trim_start_frames
    segment.trim_end_frames = body.trim_end_frames
    await db.commit()
    await db.refresh(segment)
    return segment


@router.get("/segments/{segment_id}/frames", response_model=FramePreviewResponse)
async def get_segment_frames(
    segment_id: UUID,
    position: str = Query(..., pattern="^(start|end)$"),
    count: int = Query(5, ge=1, le=20),
    trim: int = Query(0, ge=0),
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
    if not segment.output_path:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Segment has no output video")

    job = await db.get(Job, segment.job_id)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        video_path = tmppath / "segment.mp4"
        await asyncio.to_thread(download_file, segment.output_path, str(video_path))

        # Get total frame count via ffprobe
        probe_cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-count_frames",
            "-show_entries", "stream=nb_read_frames,r_frame_rate",
            "-of", "json",
            str(video_path),
        ]
        probe = await asyncio.to_thread(subprocess.run, probe_cmd, capture_output=True, timeout=60)
        if probe.returncode != 0:
            raise HTTPException(status_code=500, detail="ffprobe failed")
        probe_data = json.loads(probe.stdout)
        stream = probe_data["streams"][0]
        total_frames = int(stream["nb_read_frames"])
        r_rate = stream["r_frame_rate"]
        num, den = r_rate.split("/")
        fps = float(num) / float(den)

        # Determine frame range centered on the trim cut point
        count = min(count, total_frames)
        if position == "start":
            # Cut point is at frame index `trim` — center around it
            cut = min(trim, total_frames - 1)
            half = count // 2
            start_frame = max(cut - half, 0)
            end_frame = min(start_frame + count - 1, total_frames - 1)
            start_frame = max(end_frame - count + 1, 0)
        else:
            # Cut point is at frame index `total_frames - trim` — center around it
            cut = max(total_frames - trim, 0)
            half = count // 2
            start_frame = max(cut - half, 0)
            end_frame = min(start_frame + count - 1, total_frames - 1)
            start_frame = max(end_frame - count + 1, 0)

        # Extract frames
        extract_cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vf", f"select='between(n\\,{start_frame}\\,{end_frame})',scale=320:-1",
            "-vsync", "vfr",
            str(tmppath / "frame_%03d.jpg"),
        ]
        extract = await asyncio.to_thread(subprocess.run, extract_cmd, capture_output=True, timeout=60)
        if extract.returncode != 0:
            raise HTTPException(status_code=500, detail="ffmpeg frame extraction failed")

        # Build response
        frames = []
        for i in range(count):
            frame_path = tmppath / f"frame_{i+1:03d}.jpg"
            if not frame_path.exists():
                break
            b64 = base64.b64encode(frame_path.read_bytes()).decode()
            frames.append(FramePreview(
                frame_index=start_frame + i,
                data_url=f"data:image/jpeg;base64,{b64}",
            ))

        return FramePreviewResponse(total_frames=total_frames, fps=fps, frames=frames)


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
    if segment.status != SegmentStatus.FAILED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Only failed segments can be retried (current: '{segment.status}')",
        )

    segment.status = SegmentStatus.PENDING
    segment.worker_id = None
    segment.worker_name = None
    segment.claimed_at = None
    segment.completed_at = None
    segment.error_message = None
    segment.progress_log = None

    job = await db.get(Job, segment.job_id)
    job.status = JobStatus.PENDING

    await db.commit()
    await db.refresh(segment)
    return segment


@router.post("/segments/{segment_id}/cancel", response_model=SegmentResponse)
async def cancel_segment(
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
    if segment.status not in (SegmentStatus.PENDING, SegmentStatus.CLAIMED, SegmentStatus.PROCESSING):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Only pending, claimed, or processing segments can be cancelled (current: '{segment.status}')",
        )

    segment.status = SegmentStatus.FAILED
    segment.error_message = "Cancelled by user"
    segment.completed_at = datetime.now(timezone.utc)
    segment.worker_id = None
    segment.worker_name = None
    segment.claimed_at = None

    job = await db.get(Job, segment.job_id)
    active_result = await db.execute(
        select(Segment).where(
            Segment.job_id == job.id,
            Segment.status.in_([SegmentStatus.PENDING, SegmentStatus.CLAIMED, SegmentStatus.PROCESSING]),
        )
    )
    if not active_result.scalars().all():
        job.status = JobStatus.AWAITING

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
    if segment.status not in (SegmentStatus.FAILED, SegmentStatus.COMPLETED):
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

    # Re-index remaining segments (use negative temp values to avoid unique constraint conflicts)
    remaining_result = await db.execute(
        select(Segment).where(Segment.job_id == job.id).order_by(Segment.index)
    )
    remaining = remaining_result.scalars().all()
    old_indices = {seg.id: seg.index for seg in remaining}
    for i, seg in enumerate(remaining):
        seg.index = -(i + 1)
    await db.flush()
    for i, seg in enumerate(remaining):
        seg.index = i

    # Rename S3 files for segments whose index changed
    for seg in remaining:
        old_idx = old_indices[seg.id]
        if old_idx == seg.index:
            continue
        for attr in ("output_path", "last_frame_path"):
            old_path = getattr(seg, attr)
            if not old_path:
                continue
            try:
                bucket, old_key = parse_s3_uri(old_path)
                new_key = old_key.replace(f"/{old_idx}_", f"/{seg.index}_", 1)
                if new_key != old_key:
                    await asyncio.to_thread(move_object, bucket, old_key, new_key)
                    setattr(seg, attr, f"s3://{bucket}/{new_key}")
            except Exception:
                logger.warning("Failed to rename S3 object for segment %s: %s", seg.id, old_path)

    # Update job status if needed
    has_failed = await db.execute(
        select(Segment).where(Segment.job_id == job.id, Segment.status == SegmentStatus.FAILED)
    )
    if job.status == JobStatus.FAILED and has_failed.scalar_one_or_none() is None:
        job.status = JobStatus.AWAITING

    await db.commit()
