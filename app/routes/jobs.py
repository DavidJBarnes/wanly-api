import asyncio
import json
import os
import random
from datetime import datetime, timezone
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Query, UploadFile, status
from sqlalchemy import case, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.auth import get_current_user
from app.database import get_db
from app.models import Job, Lora, Segment, User, Video
from app.routes.segments import _resolve_loras, _resolve_wildcards
from app.config import settings
from app.s3 import delete_object, delete_prefix, upload_bytes
from app.schemas.jobs import JobCreate, JobDetailResponse, JobListResponse, JobReorderRequest, JobResponse, JobUpdate, StatsResponse, WorkerStatsItem
from app.stitch import stitch_video

import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/jobs", response_model=JobResponse, status_code=status.HTTP_201_CREATED)
async def create_job(
    data: str = Form(...),
    starting_image: UploadFile | None = File(None),
    faceswap_image: UploadFile | None = File(None),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    try:
        body = JobCreate.model_validate_json(data)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"Invalid JSON in data field: {e}")

    seed = body.seed if body.seed is not None else random.randint(0, 2**63 - 1)

    # New jobs go to bottom of queue
    max_priority_result = await db.execute(
        select(func.coalesce(func.max(Job.priority), -1)).where(Job.user_id == user.id)
    )
    next_priority = max_priority_result.scalar_one() + 1

    job = Job(
        user_id=user.id,
        name=body.name,
        width=body.width,
        height=body.height,
        fps=body.fps,
        seed=seed,
        priority=next_priority,
    )
    db.add(job)
    await db.flush()  # Get job.id

    # Upload starting image to S3 if provided
    if starting_image is not None:
        image_data = await starting_image.read()
        ext = os.path.splitext(starting_image.filename or "image.png")[1] or ".png"
        key = f"{job.id}/starting_image{ext}"
        uri = await asyncio.to_thread(upload_bytes, image_data, key, settings.s3_jobs_bucket)
        job.starting_image = uri

    # Upload faceswap image to S3 if provided
    faceswap_uri = None
    if faceswap_image is not None:
        fs_data = await faceswap_image.read()
        ext = os.path.splitext(faceswap_image.filename or "face.png")[1] or ".png"
        key = f"{job.id}/faceswap_source{ext}"
        faceswap_uri = await asyncio.to_thread(upload_bytes, fs_data, key, settings.s3_jobs_bucket)

    seg = body.first_segment
    resolved_loras = await _resolve_loras(db, seg.loras)
    resolved_prompt, prompt_template = await _resolve_wildcards(db, seg.prompt)
    segment = Segment(
        job_id=job.id,
        index=0,
        prompt=resolved_prompt,
        prompt_template=prompt_template,
        duration_seconds=seg.duration_seconds,
        speed=seg.speed,
        start_image=seg.start_image,
        loras=resolved_loras,
        faceswap_enabled=seg.faceswap_enabled,
        faceswap_method=seg.faceswap_method,
        faceswap_source_type=seg.faceswap_source_type,
        faceswap_image=faceswap_uri if faceswap_uri else seg.faceswap_image,
        faceswap_faces_order=seg.faceswap_faces_order,
        faceswap_faces_index=seg.faceswap_faces_index,
        auto_finalize=seg.auto_finalize,
    )
    db.add(segment)
    await db.commit()
    await db.refresh(job)
    return job


@router.get("/jobs", response_model=JobListResponse)
async def list_jobs(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    status_filter: str | None = Query(None, alias="status"),
    sort: str = Query("created_at_desc"),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    base = select(Job).where(Job.user_id == user.id)
    if status_filter:
        statuses = [s.strip() for s in status_filter.split(",") if s.strip()]
        base = base.where(Job.status.in_(statuses))
    else:
        base = base.where(Job.status.notin_(["finalized", "finalizing"]))

    total_result = await db.execute(select(func.count()).select_from(base.subquery()))
    total = total_result.scalar_one()

    if sort == "priority_asc":
        order = [Job.priority.asc(), Job.created_at.asc()]
    else:
        order = [Job.created_at.desc()]

    result = await db.execute(
        base.order_by(*order).offset(offset).limit(limit)
    )
    items = list(result.scalars().all())

    # Aggregate segment counts per job in a single query
    job_ids = [j.id for j in items]
    counts_map: dict[UUID, tuple[int, int]] = {}
    if job_ids:
        counts_result = await db.execute(
            select(
                Segment.job_id,
                func.count().label("total"),
                func.count(case((Segment.status == "completed", 1))).label("completed"),
            )
            .where(Segment.job_id.in_(job_ids))
            .group_by(Segment.job_id)
        )
        for row in counts_result.all():
            counts_map[row[0]] = (row[1], row[2])

    response_items = []
    for j in items:
        seg_total, seg_completed = counts_map.get(j.id, (0, 0))
        response_items.append(
            JobResponse(
                id=j.id,
                name=j.name,
                width=j.width,
                height=j.height,
                fps=j.fps,
                seed=j.seed,
                starting_image=j.starting_image,
                priority=j.priority,
                status=j.status,
                segment_count=seg_total,
                completed_segment_count=seg_completed,
                created_at=j.created_at,
                updated_at=j.updated_at,
            )
        )

    return JobListResponse(items=response_items, total=total, limit=limit, offset=offset)


@router.put("/jobs/reorder", response_model=list[JobResponse])
async def reorder_jobs(
    body: JobReorderRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if not body.job_ids:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="job_ids must not be empty")

    # Fetch all referenced jobs, verify they belong to the user
    result = await db.execute(
        select(Job).where(Job.id.in_(body.job_ids), Job.user_id == user.id)
    )
    jobs_by_id = {job.id: job for job in result.scalars().all()}

    if len(jobs_by_id) != len(body.job_ids):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Some job IDs not found or not owned by you")

    # Assign priority 0, 1, 2, ... based on array position
    for i, job_id in enumerate(body.job_ids):
        jobs_by_id[job_id].priority = i

    await db.commit()

    # Return in priority order
    ordered = [jobs_by_id[jid] for jid in body.job_ids]
    for job in ordered:
        await db.refresh(job)
    return ordered


@router.get("/jobs/{job_id}", response_model=JobDetailResponse)
async def get_job(
    job_id: UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Job)
        .where(Job.id == job_id, Job.user_id == user.id)
        .options(selectinload(Job.segments), selectinload(Job.videos))
    )
    job = result.scalar_one_or_none()
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")

    segments = job.segments
    completed = [s for s in segments if s.status == "completed"]
    total_run_time = 0.0
    for s in completed:
        if s.claimed_at and s.completed_at:
            total_run_time += (s.completed_at - s.claimed_at).total_seconds()
    total_video_time = sum(s.duration_seconds for s in completed)

    return JobDetailResponse(
        id=job.id,
        name=job.name,
        width=job.width,
        height=job.height,
        fps=job.fps,
        seed=job.seed,
        starting_image=job.starting_image,
        priority=job.priority,
        status=job.status,
        created_at=job.created_at,
        updated_at=job.updated_at,
        segments=segments,
        videos=job.videos,
        segment_count=len(segments),
        completed_segment_count=len(completed),
        total_run_time=total_run_time,
        total_video_time=total_video_time,
    )


@router.patch("/jobs/{job_id}", response_model=JobResponse)
async def update_job(
    job_id: UUID,
    body: JobUpdate,
    background_tasks: BackgroundTasks,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Job).where(Job.id == job_id, Job.user_id == user.id))
    job = result.scalar_one_or_none()
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")

    if body.name is not None:
        job.name = body.name

    if body.status is not None:
        valid_transitions = {
            "pending": {"paused"},
            "processing": {"paused"},
            "awaiting": {"paused", "finalized"},
            "failed": {"paused"},
            "paused": {"pending", "processing", "awaiting"},
        }
        allowed = valid_transitions.get(job.status, set())
        if body.status not in allowed:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot transition from '{job.status}' to '{body.status}'",
            )
        job.status = body.status
        if body.status == "finalized":
            video = Video(job_id=job.id, status="pending")
            db.add(video)
            await db.flush()
            background_tasks.add_task(stitch_video, video.id, job.id)

    await db.commit()
    await db.refresh(job)
    return job


@router.post("/jobs/{job_id}/reopen", response_model=JobDetailResponse)
async def reopen_job(
    job_id: UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Job)
        .where(Job.id == job_id, Job.user_id == user.id)
        .options(selectinload(Job.segments), selectinload(Job.videos))
    )
    job = result.scalar_one_or_none()
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    if job.status != "finalized":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only finalized jobs can be re-opened",
        )

    # Delete video records and their S3 objects
    for video in job.videos:
        if video.output_path:
            try:
                await asyncio.to_thread(delete_object, video.output_path)
            except Exception:
                logger.warning("Failed to delete S3 object %s", video.output_path, exc_info=True)
        await db.delete(video)

    job.status = "awaiting"
    await db.commit()
    await db.refresh(job, attribute_names=["segments", "videos"])

    segments = job.segments
    completed = [s for s in segments if s.status == "completed"]
    total_run_time = 0.0
    for s in completed:
        if s.claimed_at and s.completed_at:
            total_run_time += (s.completed_at - s.claimed_at).total_seconds()
    total_video_time = sum(s.duration_seconds for s in completed)

    return JobDetailResponse(
        id=job.id, name=job.name, width=job.width, height=job.height,
        fps=job.fps, seed=job.seed, starting_image=job.starting_image,
        priority=job.priority, status=job.status,
        created_at=job.created_at, updated_at=job.updated_at,
        segments=segments, videos=job.videos,
        segment_count=len(segments), completed_segment_count=len(completed),
        total_run_time=total_run_time, total_video_time=total_video_time,
    )


@router.delete("/jobs/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_job(
    job_id: UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Job)
        .where(Job.id == job_id, Job.user_id == user.id)
        .options(selectinload(Job.segments), selectinload(Job.videos))
    )
    job = result.scalar_one_or_none()
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")

    if job.status in ("processing", "finalizing"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot delete a job that is currently {job.status}",
        )

    # Best-effort S3 cleanup — delete all objects under the job prefix
    try:
        deleted = await asyncio.to_thread(delete_prefix, f"{job_id}/", settings.s3_jobs_bucket)
        logger.info("Deleted %d S3 objects for job %s", deleted, job_id)
    except Exception:
        logger.warning("Failed to delete S3 objects for job %s", job_id, exc_info=True)

    # Delete DB records in FK order
    for video in job.videos:
        await db.delete(video)
    for segment in job.segments:
        await db.delete(segment)
    await db.delete(job)
    await db.commit()


@router.get("/stats", response_model=StatsResponse)
async def get_stats(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    # Jobs grouped by status
    job_rows = (
        await db.execute(
            select(Job.status, func.count())
            .where(Job.user_id == user.id)
            .group_by(Job.status)
        )
    ).all()
    jobs_by_status = {row[0]: row[1] for row in job_rows}

    # Segments grouped by status (join through jobs for user scoping)
    seg_rows = (
        await db.execute(
            select(Segment.status, func.count())
            .join(Job, Segment.job_id == Job.id)
            .where(Job.user_id == user.id)
            .group_by(Segment.status)
        )
    ).all()
    segments_by_status = {row[0]: row[1] for row in seg_rows}

    # Avg run time and totals for completed segments
    agg = (
        await db.execute(
            select(
                func.avg(
                    func.extract("epoch", Segment.completed_at)
                    - func.extract("epoch", Segment.claimed_at)
                ),
                func.count(),
                func.coalesce(func.sum(Segment.duration_seconds), 0),
            )
            .join(Job, Segment.job_id == Job.id)
            .where(Job.user_id == user.id, Segment.status == "completed")
        )
    ).one()
    avg_run_time = round(agg[0], 1) if agg[0] is not None else None
    total_completed = agg[1]
    total_video_time = float(agg[2])

    # Worker stats
    worker_rows = (
        await db.execute(
            select(
                Segment.worker_name,
                func.count(),
                func.avg(
                    func.extract("epoch", Segment.completed_at)
                    - func.extract("epoch", Segment.claimed_at)
                ),
                func.max(Segment.completed_at),
            )
            .join(Job, Segment.job_id == Job.id)
            .where(
                Job.user_id == user.id,
                Segment.status == "completed",
                Segment.worker_name.isnot(None),
            )
            .group_by(Segment.worker_name)
        )
    ).all()
    worker_stats = [
        WorkerStatsItem(
            worker_name=row[0],
            segments_completed=row[1],
            avg_run_time=round(row[2], 1) if row[2] else 0,
            last_seen=row[3],
        )
        for row in worker_rows
    ]

    return StatsResponse(
        jobs_by_status=jobs_by_status,
        segments_by_status=segments_by_status,
        avg_segment_run_time=avg_run_time,
        total_segments_completed=total_completed,
        total_video_time=total_video_time,
        worker_stats=worker_stats,
    )
