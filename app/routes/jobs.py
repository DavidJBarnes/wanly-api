import asyncio
import json
import os
import random
from datetime import datetime, timezone
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.auth import get_current_user
from app.database import get_db
from app.models import Job, Lora, Segment, User, Video
from app.routes.segments import _resolve_loras, _resolve_wildcards
from app.config import settings
from app.s3 import upload_bytes
from app.schemas.jobs import JobCreate, JobDetailResponse, JobListResponse, JobResponse, JobUpdate, StatsResponse, WorkerStatsItem

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
    job = Job(
        user_id=user.id,
        name=body.name,
        width=body.width,
        height=body.height,
        fps=body.fps,
        seed=seed,
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
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    base = select(Job).where(Job.user_id == user.id)
    if status_filter:
        statuses = [s.strip() for s in status_filter.split(",") if s.strip()]
        base = base.where(Job.status.in_(statuses))

    total_result = await db.execute(select(func.count()).select_from(base.subquery()))
    total = total_result.scalar_one()

    result = await db.execute(
        base.order_by(Job.created_at.desc()).offset(offset).limit(limit)
    )
    items = result.scalars().all()

    return JobListResponse(items=items, total=total, limit=limit, offset=offset)


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

    await db.commit()
    await db.refresh(job)
    return job


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
