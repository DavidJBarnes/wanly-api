import random
from datetime import datetime, timezone
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.auth import get_current_user
from app.database import get_db
from app.models import Job, Segment, User, Video
from app.schemas.jobs import JobCreate, JobDetailResponse, JobResponse, JobUpdate

router = APIRouter()


@router.post("/jobs", response_model=JobResponse, status_code=status.HTTP_201_CREATED)
async def create_job(
    body: JobCreate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    seed = body.seed if body.seed is not None else random.randint(0, 2**63 - 1)
    job = Job(
        user_id=user.id,
        name=body.name,
        width=body.width,
        height=body.height,
        fps=body.fps,
        seed=seed,
        starting_image=body.starting_image,
    )
    db.add(job)
    await db.flush()

    seg = body.first_segment
    segment = Segment(
        job_id=job.id,
        index=0,
        prompt=seg.prompt,
        duration_seconds=seg.duration_seconds,
        start_image=seg.start_image,
        loras=seg.loras,
        faceswap_enabled=seg.faceswap_enabled,
        faceswap_method=seg.faceswap_method,
        faceswap_source_type=seg.faceswap_source_type,
        faceswap_image=seg.faceswap_image,
        faceswap_faces_order=seg.faceswap_faces_order,
        faceswap_faces_index=seg.faceswap_faces_index,
        auto_finalize=seg.auto_finalize,
    )
    db.add(segment)
    await db.commit()
    await db.refresh(job)
    return job


@router.get("/jobs", response_model=list[JobResponse])
async def list_jobs(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Job).where(Job.user_id == user.id).order_by(Job.created_at.desc())
    )
    return result.scalars().all()


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
