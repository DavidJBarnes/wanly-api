import asyncio
import hashlib
import json
import os
import random
import re
from datetime import datetime, timedelta, timezone
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Query, UploadFile, status
from sqlalchemy import case, func, select, true
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from sqlalchemy import or_

from app.auth import get_current_user
from app.config import settings
from app.database import get_db
from app.enums import JOB_VALID_TRANSITIONS, JobStatus, SegmentStatus, VideoStatus
from app.seeds import new_seed
from app.estimation import estimate_segment_time, get_estimation_rates, estimate_queue_drain_seconds
from app.models import Job, Segment, User, Video
from app.routes.segments import _resolve_wildcards
from app.s3 import delete_object, delete_prefix, delete_prefix_except, upload_bytes
from app.tag_filter import like_escape, tag_clause

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ALLOWED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}


def _image_ext(filename: str | None) -> str:
    ext = os.path.splitext(filename or "")[1].lower()
    return ext if ext in _ALLOWED_IMAGE_EXTS else ".png"


def _starting_image_key(user_id: UUID, image_hash: str, ext: str) -> str:
    return f"users/{user_id}/starting_images/{image_hash}{ext}"
from app.schemas.jobs import JobCreate, JobDetailResponse, JobListResponse, JobReorderRequest, JobResponse, JobUpdate, StatsResponse, WorkerStatsItem
from app.schemas.segments import SegmentResponse
from app.stitch import stitch_video

import logging

logger = logging.getLogger(__name__)

router = APIRouter()




@router.get("/jobs/starting-image-exists")
async def starting_image_exists(
    sha256: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Check whether this user has already uploaded a starting image with the given SHA-256.

    The console hashes files client-side and calls this before POSTing a new job —
    if the image is already in S3 for this user, the console passes
    `starting_image_hash` in the job body instead of re-uploading the bytes.
    """
    if not _SHA256_RE.match(sha256):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid SHA-256 hash")
    result = await db.execute(
        select(Job.starting_image)
        .where(Job.user_id == user.id, Job.starting_image_hash == sha256)
        .limit(1)
    )
    uri = result.scalar_one_or_none()
    return {"exists": uri is not None, "uri": uri}


@router.post("/jobs", response_model=JobResponse, status_code=status.HTTP_201_CREATED)
async def create_job(
    data: str = Form(...),
    starting_image: UploadFile | None = File(None),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    try:
        body = JobCreate.model_validate_json(data)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"Invalid JSON in data field: {e}")

    # Drawn below 2**53, the largest integer JavaScript represents exactly.
    #
    # The seed is displayed on the job page and can be typed back into this dialog to reproduce a
    # job — that round trip is the only reason to show it. Above 2**53 the browser silently rounds
    # it, so the number on screen is NOT the number that generated the video and reproducing from
    # it quietly produces something else. Drawing from the full BigInteger range made that near
    # certain rather than rare: only about one seed in a thousand landed low enough to survive.
    #
    # Jobs created before this keep their large seeds; nothing can recover what those displayed.
    seed = body.seed if body.seed is not None else new_seed()

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
        continuation_mode=body.continuation_mode,
        # Lynx engine selection + tunables. All optional: None -> daemon settings default.
        generation_engine=body.generation_engine,
        lynx_subject_image=body.lynx_subject_image,
        lynx_ip_scale=body.lynx_ip_scale,
        lynx_ref_scale=body.lynx_ref_scale,
        lynx_cfg_scale=body.lynx_cfg_scale,
        lynx_start_percent=body.lynx_start_percent,
        lynx_end_percent=body.lynx_end_percent,
        lynx_ref_blocks_to_use=body.lynx_ref_blocks_to_use,
        lynx_ip_layers=body.lynx_ip_layers,
        lynx_resampler=body.lynx_resampler,
        lynx_steps=body.lynx_steps,
        lynx_cfg=body.lynx_cfg,
        lynx_shift=body.lynx_shift,
        lynx_scheduler=body.lynx_scheduler,
        lynx_distill_strength=body.lynx_distill_strength,
        priority=next_priority,
        tags=body.tags,
    )
    db.add(job)
    await db.flush()  # Get job.id

    # Upload starting image to S3 if provided (with hash-based storage + bandwidth dedup)
    if starting_image is not None:
        image_data = await starting_image.read()
        image_hash = hashlib.sha256(image_data).hexdigest()

        # If this user has already uploaded this exact image, reuse its URI.
        existing = await db.execute(
            select(Job.starting_image)
            .where(Job.user_id == user.id, Job.starting_image_hash == image_hash)
            .limit(1)
        )
        existing_uri = existing.scalar_one_or_none()

        if existing_uri:
            job.starting_image = existing_uri
        else:
            ext = _image_ext(starting_image.filename)
            key = _starting_image_key(user.id, image_hash, ext)
            uri = await asyncio.to_thread(upload_bytes, image_data, key, settings.s3_jobs_bucket)
            job.starting_image = uri
        job.starting_image_hash = image_hash
    elif body.starting_image_hash:
        # Client already hashed the file and confirmed the server has it — skip re-upload.
        if not _SHA256_RE.match(body.starting_image_hash):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid SHA-256 hash")
        existing = await db.execute(
            select(Job.starting_image)
            .where(Job.user_id == user.id, Job.starting_image_hash == body.starting_image_hash)
            .limit(1)
        )
        existing_uri = existing.scalar_one_or_none()
        if not existing_uri:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No existing image found for that hash; upload the file instead.",
            )
        job.starting_image = existing_uri
        job.starting_image_hash = body.starting_image_hash
    elif body.starting_image_uri:
        job.starting_image = body.starting_image_uri

    # Lynx conditions identity on a subject image. The console reuses the starting-image
    # upload path for it (same crop/hash-dedup machinery), so mirror the resolved URI
    # across unless the caller set lynx_subject_image explicitly. Despite sharing the
    # upload slot it is NOT a first frame — Lynx is a T2V base, and the subject never
    # appears as frame 0.
    if job.generation_engine == "lynx" and not job.lynx_subject_image:
        job.lynx_subject_image = job.starting_image

    seg = body.first_segment
    resolved_prompt, prompt_template = await _resolve_wildcards(db, seg.prompt)
    segment = Segment(
        job_id=job.id,
        index=0,
        prompt=resolved_prompt,
        prompt_template=prompt_template,
        duration_seconds=seg.duration_seconds,
        speed=seg.speed,
        start_image=seg.start_image,
        negative_prompt=seg.negative_prompt,
        ltx_recipe=seg.ltx_recipe,
        auto_finalize=seg.auto_finalize,
    )
    db.add(segment)
    await db.commit()
    await db.refresh(job)
    return job


def job_filter(
    *,
    user_id: UUID,
    status_filter: str | None = None,
    name: str | None = None,
    search: str | None = None,
    tags: list[str] | None = None,
) -> list:
    """Every criterion ANDs. Returns clauses for .where(*clauses).

    Two controls with two different jobs, the same split the image repo settled on. `tags`
    matches a tag in FULL, so the AR pill returns AR videos and not every job whose name happens
    to contain "ar"; `q` matches the job NAME as a fragment, because a half-remembered name is
    often the only handle there is.

    `q` also matches a whole tag, so typing a tag name still finds it -- but it no longer matches
    tags by substring, which is what made searching "AR" return jobs tagged `argentina` or
    `starlight` (#353).

    Tag pills are strictly conjunctive: each one narrows, so two pills mean "both tags on one
    job". There is no OR.
    """
    clauses = [Job.user_id == user_id]
    if name:
        clauses.append(Job.name.ilike(f"%{like_escape(name)}%", escape="\\"))
    if search and search.strip():
        clauses.append(
            or_(
                Job.name.ilike(f"%{like_escape(search.strip())}%", escape="\\"),
                tag_clause(Job.tags, search),
            )
        )
    clauses += [tag_clause(Job.tags, t) for t in (tags or []) if t.strip()]
    if status_filter:
        statuses = [s.strip() for s in status_filter.split(",") if s.strip()]
        clauses.append(Job.status.in_(statuses))
    else:
        clauses.append(Job.status.notin_([JobStatus.FINALIZED, JobStatus.FINALIZING, JobStatus.ARCHIVED]))
    return clauses


@router.get("/jobs", response_model=JobListResponse)
async def list_jobs(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    status_filter: str | None = Query(None, alias="status"),
    sort: str = Query("created_at_desc"),
    name: str | None = Query(None, min_length=1, max_length=255, description="Filter jobs by name (case-insensitive partial match)"),
    search: str | None = Query(None, min_length=1, max_length=255, alias="q", description="Search jobs by name (partial) or by a whole tag"),
    tags: list[str] = Query(default_factory=list, description="Only jobs carrying ALL of these tags, matched in full"),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    base = select(Job).where(
        *job_filter(
            user_id=user.id,
            status_filter=status_filter,
            name=name,
            search=search,
            tags=tags,
        )
    )

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
            # Progress excludes discarded segments from BOTH halves. A job whose only bad
            # segment was discarded should read 4/4, not 5/5 with one of them not in the video.
            #
            # Run-time totals elsewhere deliberately still INCLUDE discarded work: the GPU time
            # was genuinely spent, and hiding it would understate what a job cost.
            select(
                Segment.job_id,
                func.count(case((Segment.discarded.is_(False), 1))).label("total"),
                func.count(case(
                    ((Segment.status == SegmentStatus.COMPLETED)
                     & Segment.discarded.is_(False), 1)
                )).label("completed"),
            )
            .where(Segment.job_id.in_(job_ids))
            .group_by(Segment.job_id)
        )
        for row in counts_result.all():
            counts_map[row[0]] = (row[1], row[2])

    # Fetch active segment info for estimation
    active_statuses = {SegmentStatus.PENDING, SegmentStatus.CLAIMED, SegmentStatus.PROCESSING}
    active_jobs = [j for j in items if j.status in (JobStatus.PENDING, JobStatus.PROCESSING)]
    active_job_ids = [j.id for j in active_jobs]
    est_map: dict[UUID, float | None] = {}
    if active_job_ids:
        active_seg_result = await db.execute(
            select(
                Segment.job_id,
                Segment.duration_seconds,
                Segment.gpu_name,
            )
            .where(
                Segment.job_id.in_(active_job_ids),
                Segment.status.in_(active_statuses),
            )
        )
        active_segs = active_seg_result.all()
        if active_segs:
            rates = await get_estimation_rates(db, user.id)
            for row in active_segs:
                seg_job_id, seg_dur, seg_gpu = row
                job_obj = next((j for j in active_jobs if j.id == seg_job_id), None)
                if job_obj:
                    est = estimate_segment_time(
                        rates, job_obj.width, job_obj.height, job_obj.fps,
                        seg_dur, seg_gpu,
                    )
                    est_map[seg_job_id] = est

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
                estimated_run_time=est_map.get(j.id),
                # Lynx engine fields. These responses are hand-built, so anything not
                # listed here is silently dropped by Pydantic even though it is on the schema.
                generation_engine=j.generation_engine,
                lynx_subject_image=j.lynx_subject_image,
                lynx_ip_scale=j.lynx_ip_scale,
                lynx_ref_scale=j.lynx_ref_scale,
                lynx_cfg_scale=j.lynx_cfg_scale,
                lynx_start_percent=j.lynx_start_percent,
                lynx_end_percent=j.lynx_end_percent,
                lynx_ref_blocks_to_use=j.lynx_ref_blocks_to_use,
                lynx_ip_layers=j.lynx_ip_layers,
                lynx_resampler=j.lynx_resampler,
                lynx_steps=j.lynx_steps,
                lynx_cfg=j.lynx_cfg,
                lynx_shift=j.lynx_shift,
                lynx_scheduler=j.lynx_scheduler,
                lynx_distill_strength=j.lynx_distill_strength,
                tags=j.tags,
                created_at=j.created_at,
                updated_at=j.updated_at,
            )
        )

    return JobListResponse(items=response_items, total=total, limit=limit, offset=offset)


@router.get("/jobs/tag-counts")
async def job_tag_counts(
    status_filter: str | None = Query(None, alias="status"),
    search: str | None = Query(None, min_length=1, max_length=255, alias="q"),
    tags: list[str] = Query(default_factory=list),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Every tag in use, with how many jobs carry it under the CURRENT filter.

    Counts scoped to the filter are what make the pills navigable rather than a guessing game:
    with `kelly` selected, the tags left standing are the ones that actually co-occur with it, so
    a dead end is visible before it is clicked instead of after. A selected tag always survives
    its own filter (its count is the result count), so it can be clicked off again.

    Driven by what is USED, not by the title_tags vocabulary -- the image repo learned that the
    hard way, where 11 in-use tags were absent from the vocabulary and pills built from it would
    have stranded them.

    Declared above `/jobs/{job_id}` on purpose: that route takes a UUID, so a path reaching it
    first would 422 rather than fall through.
    """
    clauses = job_filter(
        user_id=user.id,
        status_filter=status_filter,
        search=search,
        tags=tags,
    )

    # unnest in a LATERAL rather than the target list: a set-returning function in the select
    # list cannot then be grouped by.
    tag_rows = (
        func.unnest(func.string_to_array(Job.tags, ","))
        .table_valued("tag")
        .render_derived(name="t")
    )
    tag_expr = func.lower(func.btrim(tag_rows.c.tag))

    stmt = (
        select(tag_expr.label("tag"), func.count().label("count"))
        .select_from(Job)
        .join(tag_rows, true())
        .where(*clauses, tag_expr != "")
        .group_by(tag_expr)
        .order_by(func.count().desc(), tag_expr)
    )
    rows = (await db.execute(stmt)).all()
    return {"items": [{"tag": r.tag, "count": r.count} for r in rows]}


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
    completed = [s for s in segments if s.status == SegmentStatus.COMPLETED]
    total_run_time = 0.0
    for s in completed:
        if s.claimed_at and s.completed_at:
            total_run_time += (s.completed_at - s.claimed_at).total_seconds()
    total_video_time = sum(s.duration_seconds for s in completed)

    # Estimate run times for non-completed segments
    active_segs = [s for s in segments if s.status in (SegmentStatus.PENDING, SegmentStatus.CLAIMED, SegmentStatus.PROCESSING)]
    seg_responses = []
    job_est = None
    if active_segs:
        rates = await get_estimation_rates(db, user.id)
        for s in segments:
            sr = SegmentResponse.model_validate(s)
            if s.status in (SegmentStatus.PENDING, SegmentStatus.CLAIMED, SegmentStatus.PROCESSING):
                est = estimate_segment_time(
                    rates, job.width, job.height, job.fps,
                    s.duration_seconds, s.gpu_name,
                )
                sr.estimated_run_time = est
                if job_est is None:
                    job_est = est
            seg_responses.append(sr)
    else:
        seg_responses = [SegmentResponse.model_validate(s) for s in segments]

    return JobDetailResponse(
        id=job.id,
        name=job.name,
        width=job.width,
        height=job.height,
        fps=job.fps,
        seed=job.seed,
        starting_image=job.starting_image,
        continuation_mode=job.continuation_mode,
        # Lynx engine fields. These responses are hand-built, so anything not
        # listed here is silently dropped by Pydantic even though it is on the schema.
        generation_engine=job.generation_engine,
        lynx_subject_image=job.lynx_subject_image,
        lynx_ip_scale=job.lynx_ip_scale,
        lynx_ref_scale=job.lynx_ref_scale,
        lynx_cfg_scale=job.lynx_cfg_scale,
        lynx_start_percent=job.lynx_start_percent,
        lynx_end_percent=job.lynx_end_percent,
        lynx_ref_blocks_to_use=job.lynx_ref_blocks_to_use,
        lynx_ip_layers=job.lynx_ip_layers,
        lynx_resampler=job.lynx_resampler,
        lynx_steps=job.lynx_steps,
        lynx_cfg=job.lynx_cfg,
        lynx_shift=job.lynx_shift,
        lynx_scheduler=job.lynx_scheduler,
        lynx_distill_strength=job.lynx_distill_strength,
        priority=job.priority,
        status=job.status,
        tags=job.tags,
        estimated_run_time=job_est,
        created_at=job.created_at,
        updated_at=job.updated_at,
        segments=seg_responses,
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
    result = await db.execute(
        select(Job).where(Job.id == job_id, Job.user_id == user.id).with_for_update()
    )
    job = result.scalar_one_or_none()
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")

    if body.name is not None:
        job.name = body.name

    if body.tags is not None:
        job.tags = body.tags

    if body.status is not None:
        allowed = JOB_VALID_TRANSITIONS.get(job.status, set())
        if body.status not in allowed:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot transition from '{job.status}' to '{body.status}'",
            )
        job.status = body.status
        if body.status == JobStatus.FINALIZED:
            video = Video(job_id=job.id, status=VideoStatus.PENDING, tags=job.tags)
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
    if job.status != JobStatus.FINALIZED:
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

    job.status = JobStatus.AWAITING
    await db.commit()
    await db.refresh(job, attribute_names=["segments", "videos"])

    segments = job.segments
    completed = [s for s in segments if s.status == SegmentStatus.COMPLETED]
    total_run_time = 0.0
    for s in completed:
        if s.claimed_at and s.completed_at:
            total_run_time += (s.completed_at - s.claimed_at).total_seconds()
    total_video_time = sum(s.duration_seconds for s in completed)

    # Estimate run times for non-completed segments
    active_segs = [s for s in segments if s.status in (SegmentStatus.PENDING, SegmentStatus.CLAIMED, SegmentStatus.PROCESSING)]
    seg_responses = []
    job_est = None
    if active_segs:
        rates = await get_estimation_rates(db, user.id)
        for s in segments:
            sr = SegmentResponse.model_validate(s)
            if s.status in (SegmentStatus.PENDING, SegmentStatus.CLAIMED, SegmentStatus.PROCESSING):
                est = estimate_segment_time(
                    rates, job.width, job.height, job.fps,
                    s.duration_seconds, s.gpu_name,
                )
                sr.estimated_run_time = est
                if job_est is None:
                    job_est = est
            seg_responses.append(sr)
    else:
        seg_responses = [SegmentResponse.model_validate(s) for s in segments]

    return JobDetailResponse(
        id=job.id, name=job.name, width=job.width, height=job.height,
        fps=job.fps, seed=job.seed, starting_image=job.starting_image,
        # Lynx engine fields. These responses are hand-built, so anything not
        # listed here is silently dropped by Pydantic even though it is on the schema.
        generation_engine=job.generation_engine,
        lynx_subject_image=job.lynx_subject_image,
        lynx_ip_scale=job.lynx_ip_scale,
        lynx_ref_scale=job.lynx_ref_scale,
        lynx_cfg_scale=job.lynx_cfg_scale,
        lynx_start_percent=job.lynx_start_percent,
        lynx_end_percent=job.lynx_end_percent,
        lynx_ref_blocks_to_use=job.lynx_ref_blocks_to_use,
        lynx_ip_layers=job.lynx_ip_layers,
        lynx_resampler=job.lynx_resampler,
        lynx_steps=job.lynx_steps,
        lynx_cfg=job.lynx_cfg,
        lynx_shift=job.lynx_shift,
        lynx_scheduler=job.lynx_scheduler,
        lynx_distill_strength=job.lynx_distill_strength,
        priority=job.priority, status=job.status,
        tags=job.tags,
        estimated_run_time=job_est,
        created_at=job.created_at, updated_at=job.updated_at,
        segments=seg_responses, videos=job.videos,
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

    if job.status in (JobStatus.PROCESSING, JobStatus.FINALIZING):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot delete a job that is currently {job.status}",
        )

    # Best-effort S3 cleanup — delete objects under the job prefix, but keep any
    # still referenced by another job (legacy dedup stored starting images under
    # the uploading job's prefix; new uploads live under users/{uid}/ so don't
    # collide with this cleanup at all).
    try:
        bucket = settings.s3_jobs_bucket
        prefix = f"{job_id}/"
        prefix_uri = f"s3://{bucket}/{prefix}"
        ref_result = await db.execute(
            select(Job.starting_image).where(
                Job.user_id == user.id,
                Job.id != job_id,
                Job.starting_image.like(f"{prefix_uri}%"),
            )
        )
        referenced = {uri for uri in ref_result.scalars().all() if uri}
        if referenced:
            deleted = await asyncio.to_thread(
                delete_prefix_except, prefix, bucket, referenced
            )
        else:
            deleted = await asyncio.to_thread(delete_prefix, prefix, bucket)
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


STATS_WINDOW_HOURS = 24
# What counts as "still queued". Mirrors the job-queue view so the dashboard total and the
# per-job estimates there are derived from the same set of work. Awaiting jobs are excluded:
# they are blocked on a prompt, so their segments are not waiting on a worker.
QUEUE_JOB_STATUSES = (JobStatus.PENDING, JobStatus.PROCESSING)
QUEUE_SEGMENT_STATUSES = (
    SegmentStatus.PENDING,
    SegmentStatus.CLAIMED,
    SegmentStatus.PROCESSING,
)


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

    # Avg run time over a rolling window, not all of history. A lifetime average barely
    # moves once there are thousands of segments behind it, so it stops reflecting how the
    # current models, settings and workers are actually performing.
    cutoff = datetime.now(timezone.utc) - timedelta(hours=STATS_WINDOW_HOURS)
    avg_run_time = (
        await db.execute(
            select(
                func.avg(
                    func.extract("epoch", Segment.completed_at)
                    - func.extract("epoch", Segment.claimed_at)
                )
            )
            .join(Job, Segment.job_id == Job.id)
            .where(
                Job.user_id == user.id,
                Segment.status == SegmentStatus.COMPLETED,
                Segment.claimed_at.isnot(None),
                Segment.completed_at >= cutoff,
            )
        )
    ).scalar_one_or_none()
    avg_run_time = round(avg_run_time, 1) if avg_run_time is not None else None

    # Work still outstanding, priced with the same estimator the job queue uses so the two
    # views agree. Segments the estimator cannot price (no comparable completed run yet)
    # contribute nothing, so this reads low rather than guessing.
    queue_rows = (
        await db.execute(
            select(
                Job.width, Job.height, Job.fps, Segment.duration_seconds, Segment.gpu_name,
                # For the in-flight correction: what has already been rendered is not still
                # ahead of you.
                Segment.status, Segment.claimed_at,
            )
            .join(Job, Segment.job_id == Job.id)
            .where(
                Job.user_id == user.id,
                Job.status.in_(QUEUE_JOB_STATUSES),
                Segment.status.in_(QUEUE_SEGMENT_STATUSES),
            )
        )
    ).all()
    total_queue_time = 0.0
    if queue_rows:  # skip three estimator queries when the queue is empty
        rates = await get_estimation_rates(db, user.id)
        # Workers that can actually claim. A drained or offline worker will never take a
        # segment, so counting it would halve a number that is not going to halve.
        # Heartbeat rather than status alone: a worker that died without saying so still
        # reads "online-busy" until something reaps it.
        cutoff = datetime.now(timezone.utc) - timedelta(
            seconds=settings.heartbeat_offline_seconds
        )
        worker_count = (
            await db.execute(
                select(func.count())
                .select_from(Worker)
                .where(
                    Worker.status.notin_(("offline", "draining")),
                    Worker.last_heartbeat.isnot(None),
                    Worker.last_heartbeat >= cutoff,
                )
            )
        ).scalar() or 0
        total_queue_time = estimate_queue_drain_seconds(
            rates, queue_rows, worker_count, datetime.now(timezone.utc)
        )

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
                Segment.status == SegmentStatus.COMPLETED,
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
        avg_segment_run_time_24h=avg_run_time,
        total_queue_time=total_queue_time,
        worker_stats=worker_stats,
    )
