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

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Query, UploadFile, status
from sqlalchemy import or_, select, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.auth import get_current_user, verify_api_key, verify_api_key_or_bearer
from app.database import get_db
from app.enums import JobStatus, SegmentStatus, VideoStatus
from app.helpers import upload_faceswap_image
from app.models import AppSetting, Job, Lora, Segment, User, Video, VideoSettingsPreset, Wildcard
from app.s3 import delete_object, download_file, move_object, parse_s3_uri
from app.schemas.segments import (
    FramePreview,
    FramePreviewResponse,
    HologramRequest,
    SegmentClaimResponse,
    SegmentClipResponse,
    SegmentCreate,
    SegmentReprocessRequest,
    SegmentResponse,
    SegmentStatusUpdate,
    SegmentTrimUpdate,
    SegmentVideoPresetUpdate,
    SmashcutRequest,
    WorkerSegmentResponse,
)
from app.models import Favorite
from app.stitch import stitch_video

logger = logging.getLogger(__name__)

# AR hologram work rides a dedicated carrier segment at this sentinel index (far above any
# real segment count), so the real video segments and the job's finalized status are never
# touched. The console hides segments at/above this index from the segment list.
HOLOGRAM_CARRIER_INDEX = 1000

# Foundry smashcut work rides its own carrier segment at this sentinel index on a dedicated
# container job (no start image, no real segments). CPU-only (ffmpeg concat), claimed on the
# same CPU track as holograms.
SMASHCUT_CARRIER_INDEX = 2000

# Reprocess types that run on the CPU-only track (no ComfyUI/GPU): holograms + smashcut concat.
_CPU_REPROCESS_TYPES = ("ar_hologram", "smashcut_concat")

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


@router.get("/segments", response_model=list[WorkerSegmentResponse], dependencies=[Depends(verify_api_key_or_bearer)])
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

    # Inherit negative_prompt from prior segment if not explicitly set.
    # Segments are eagerly loaded and ordered by index via the relationship,
    # so job.segments[-1] reliably returns the previous segment.
    negative_prompt = body.negative_prompt
    if negative_prompt is None and job.segments:
        negative_prompt = job.segments[-1].negative_prompt

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
        faceswap_model=body.faceswap_model,
        faceswap_pixel_boost=body.faceswap_pixel_boost,
        seed_faceswap=body.seed_faceswap,
        negative_prompt=negative_prompt,
        auto_finalize=body.auto_finalize,
        video_preset_id=body.video_preset_id,
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
    kind: str = Query(None, description="'gpu' (exclude holograms), 'hologram' (only holograms), or None (any)"),
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

    # Split work by kind so the CPU-only reprocess track runs concurrently with GPU generation:
    #   kind="gpu"      -> generations only (exclude CPU reprocess carriers)
    #   kind="hologram" -> CPU-only reprocess carriers: holograms + smashcut (claimable even on
    #                      FINALIZED/FINALIZING jobs)
    #   kind=None       -> either (legacy combined behavior)
    if kind == "hologram":
        where = (
            Segment.status == SegmentStatus.PENDING,
            Segment.reprocess_type.in_(_CPU_REPROCESS_TYPES),
        )
    elif kind == "gpu":
        where = (
            Segment.status == SegmentStatus.PENDING,
            Job.status.in_([JobStatus.PENDING, JobStatus.PROCESSING]),
            # real generations (reprocess_type NULL) + GPU reprocess (faceswap); exclude CPU carriers.
            # NULL NOT IN (...) is NULL (excludes), so OR the NULL case in explicitly.
            or_(
                Segment.reprocess_type.is_(None),
                Segment.reprocess_type.not_in(_CPU_REPROCESS_TYPES),
            ),
        )
    else:
        where = (
            Segment.status == SegmentStatus.PENDING,
            or_(
                Job.status.in_([JobStatus.PENDING, JobStatus.PROCESSING]),
                Segment.reprocess_type.in_(_CPU_REPROCESS_TYPES),
            ),
        )
    result = await db.execute(
        select(Segment)
        .join(Job, Segment.job_id == Job.id)
        .where(*where)
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
    previous_segment = None
    previous_motion_keywords = None
    previous_motion_magnitude = None
    reference_frames = []
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
                previous_segment = prev_segment
                previous_motion_keywords = prev_segment.motion_keywords
                previous_motion_magnitude = prev_segment.motion_magnitude
                if prev_segment.reference_frames:
                    reference_frames = prev_segment.reference_frames.copy()
                if prev_segment.last_frame_path and prev_segment.last_frame_path not in reference_frames:
                    reference_frames.append(prev_segment.last_frame_path)
                    reference_frames = reference_frames[-3:]

    # Use segment negative_prompt if set, otherwise fall back to global app setting
    if segment.negative_prompt is not None:
        negative_prompt = segment.negative_prompt
    else:
        neg_setting = await db.get(AppSetting, "negative_prompt")
        negative_prompt = neg_setting.value if neg_setting else None

    # Resolve continuation mode API-side (VACE vs traditional). seg0 is always i2v;
    # VACE requires index>0 + the previous segment's video. The daemon falls back to
    # traditional if it isn't VACE-capable, so flipping this is safe.
    mode_setting = await db.get(AppSetting, "continuation_mode")
    global_mode = mode_setting.value if mode_setting else "traditional"
    overlap_setting = await db.get(AppSetting, "vace_overlap_frames")
    vace_overlap = int(overlap_setting.value) if overlap_setting else 12
    effective_mode = job.continuation_mode or global_mode
    prev_output_path = previous_segment.output_path if previous_segment else None
    # VACE's activation memory doesn't fit above ~480p on a 24GB card — 720p OOMs at the
    # sampler (block-swap only offloads weights, not activations). Cap VACE by resolution;
    # larger frames fall back to the traditional hard-cut continuation.
    vace_fits_vram = max(job.width, job.height) <= 896
    # VACE neutralized 2026-07-14: traditional i2v continuation is the validated path.
    # VACE runs a T2V base + always-distilled sampler, so it can't use the i2v motion/
    # identity recipe — it's a weak identity lever (see identity-drift audit). Kept dormant
    # (all the build code/columns remain); flip _VACE_ENABLED back to True to re-activate.
    _VACE_ENABLED = False
    use_vace = (
        _VACE_ENABLED
        and effective_mode == "vace"
        and segment.index > 0
        and prev_output_path is not None
        and vace_fits_vram
    )
    continuation_mode = "vace" if use_vace else "traditional"

    # Seed re-anchor: faceswap this segment's last frame to its own faceswap face before the
    # frame seeds the next segment. Author-set per segment.
    #
    # This used to be a global app setting gated on "a later segment already exists". That
    # gate could never pass: a job is created with segment 0 only and continuations are
    # appended after it runs, so at claim time there is never a successor. The feature was
    # dead in every job. No gate now — if the author asked for it, it fires. The cost when
    # the segment turns out to be last is one wasted still-image faceswap (seconds).
    seed_faceswap = bool(segment.seed_faceswap)

    # Resolve video (sampler) settings: segment's preset -> job's preset -> job's raw params.
    # Live-linked, so editing a preset changes future claims that reference it.
    vp_id = segment.video_preset_id or job.video_preset_id
    vsettings = job
    effective_loras = segment.loras
    if vp_id is not None:
        preset = await db.get(VideoSettingsPreset, vp_id)
        if preset is not None:
            vsettings = preset
            # A full-recipe preset owns its LoRAs (resolved live); a sampler-only preset
            # leaves the segment's own LoRAs untouched.
            if preset.loras:
                effective_loras = await _resolve_loras(db, preset.loras)
    # Sampler/scheduler only come from a preset (empty -> daemon default euler/simple).
    sampler_name = getattr(vsettings, "sampler_name", None) or None
    scheduler = getattr(vsettings, "scheduler", None) or None

    # AR hologram carrier: the source is the job's finalized stitched video, not this
    # segment's own output. The daemon mattes/packs it into a color+alpha hologram.
    hologram_source_path = None
    if segment.reprocess_type == "ar_hologram":
        holo_video = (
            await db.execute(
                select(Video)
                .where(Video.job_id == job.id, Video.status == VideoStatus.COMPLETED)
                .order_by(Video.created_at.desc())
            )
        ).scalars().first()
        hologram_source_path = holo_video.output_path if holo_video else None

    await db.commit()
    await db.refresh(segment)

    is_smashcut = segment.reprocess_type == "smashcut_concat"
    smashcut_clip_paths = segment.smashcut_clip_paths if is_smashcut else None
    smashcut_clip_speeds = segment.smashcut_clip_speeds if is_smashcut else None

    return SegmentClaimResponse(
        id=segment.id,
        job_id=segment.job_id,
        index=segment.index,
        prompt=segment.prompt,
        duration_seconds=segment.duration_seconds,
        speed=segment.speed,
        start_image=resolved_start_image,
        loras=effective_loras,
        faceswap_enabled=segment.faceswap_enabled,
        faceswap_method=segment.faceswap_method,
        faceswap_source_type=segment.faceswap_source_type,
        faceswap_image=segment.faceswap_image,
        faceswap_faces_order=segment.faceswap_faces_order,
        faceswap_faces_index=segment.faceswap_faces_index,
        faceswap_model=segment.faceswap_model,
        faceswap_pixel_boost=segment.faceswap_pixel_boost,
        initial_reference_image=job.identity_reference_image or job.starting_image,
        # Ground truth for identity scoring is always segment 0's start frame. No override:
        # "her" is defined by where the job began, not by a separate reference image.
        identity_ground_truth=job.starting_image,
        motion_keywords=segment.motion_keywords,
        previous_motion_keywords=previous_motion_keywords,
        previous_motion_magnitude=previous_motion_magnitude,
        reference_frames=reference_frames if reference_frames else None,
        lightx2v_strength_high=vsettings.lightx2v_strength_high,
        lightx2v_strength_low=vsettings.lightx2v_strength_low,
        cfg_high=vsettings.cfg_high,
        cfg_low=vsettings.cfg_low,
        steps_total=vsettings.steps_total,
        high_noise_steps=vsettings.high_noise_steps,
        flow_shift=vsettings.flow_shift,
        sampler_name=sampler_name,
        scheduler=scheduler,
        negative_prompt=negative_prompt,
        reprocess_type=segment.reprocess_type,
        output_path=segment.output_path,
        seed_faceswap=seed_faceswap,
        width=job.width,
        height=job.height,
        fps=job.fps,
        # Per-segment seed = job.seed + index: seg0 stays exactly job.seed (reproducible),
        # later segments get decorrelated noise so they don't repeat the same motion pattern.
        # Whole job still reproduces identically from job.seed. Modulo keeps it in range.
        seed=(job.seed + segment.index) % (2**63 - 1),
        continuation_mode=continuation_mode,
        previous_output_path=prev_output_path if use_vace else None,
        vace_overlap_frames=vace_overlap,
        hologram_source_path=hologram_source_path,
        hologram_key_color=segment.hologram_key_color,
        hologram_subject_height_m=segment.hologram_subject_height_m,
        hologram_flavor=segment.hologram_flavor,
        hologram_depth_scale_m=segment.hologram_depth_scale_m,
        smashcut_clip_paths=smashcut_clip_paths,
        smashcut_transition=segment.smashcut_transition,
        smashcut_clip_speeds=smashcut_clip_speeds,
        # Lynx tunables live on the job (they describe the whole generation, not one
        # segment) and are passed through verbatim: None means "daemon settings default".
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
    if body.motion_keywords is not None:
        segment.motion_keywords = body.motion_keywords
    if body.motion_magnitude is not None:
        segment.motion_magnitude = body.motion_magnitude
    # Identity scores measured by the daemon at generation time. Measurement only -
    # never gates status, exactly like motion_magnitude and the Lynx QA scores.
    if body.identity_mean_cos is not None:
        segment.identity_mean_cos = body.identity_mean_cos
    if body.identity_mean_cos_ref is not None:
        segment.identity_mean_cos_ref = body.identity_mean_cos_ref
    if body.identity_min_cos is not None:
        segment.identity_min_cos = body.identity_min_cos
    if body.identity_slope is not None:
        segment.identity_slope = body.identity_slope
    if body.identity_frames is not None:
        segment.identity_frames = body.identity_frames
    if body.identity_no_face is not None:
        segment.identity_no_face = body.identity_no_face
    if body.identity_face_px_p50 is not None:
        segment.identity_face_px_p50 = body.identity_face_px_p50
    if body.identity_yaw_max is not None:
        segment.identity_yaw_max = body.identity_yaw_max
    if body.identity_metrics is not None:
        segment.identity_metrics = body.identity_metrics
    if body.identity_start_cos_ref is not None:
        segment.identity_start_cos_ref = body.identity_start_cos_ref
    if body.identity_end_cos_ref is not None:
        segment.identity_end_cos_ref = body.identity_end_cos_ref
    if body.vace_overlap_seconds is not None:
        segment.vace_overlap_seconds = body.vace_overlap_seconds
    if body.lynx_identity_scores is not None:
        # Identity QA is measurement only — persisted so the ip/ref calibration can be
        # argued from numbers later. Never affects segment status.
        segment.lynx_identity_scores = body.lynx_identity_scores
        logger.info(
            "segment %s lynx identity QA: mean=%s min=%s max=%s (%s/%s frames with a face)",
            segment.id,
            body.lynx_identity_scores.get("mean"),
            body.lynx_identity_scores.get("min"),
            body.lynx_identity_scores.get("max"),
            body.lynx_identity_scores.get("frames_with_face"),
            body.lynx_identity_scores.get("frames_sampled"),
        )

    await db.flush()

    # Check if job needs status update. Hologram carriers are exempt — they don't affect the
    # source job's status (a failed hologram must not flip a finalized job to FAILED).
    if body.status in (SegmentStatus.COMPLETED, SegmentStatus.FAILED) and segment.reprocess_type != "ar_hologram":
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
                video = Video(job_id=job.id, status=VideoStatus.PENDING, tags=job.tags)
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


@router.patch("/segments/{segment_id}/video-preset", response_model=SegmentResponse)
async def update_segment_video_preset(
    segment_id: UUID,
    body: SegmentVideoPresetUpdate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Set/clear a segment's video-settings preset override (applies to the next claim)."""
    result = await db.execute(
        select(Segment)
        .join(Job, Segment.job_id == Job.id)
        .where(Segment.id == segment_id, Job.user_id == user.id)
    )
    segment = result.scalar_one_or_none()
    if segment is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Segment not found")
    segment.video_preset_id = body.video_preset_id
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


@router.post("/segments/{segment_id}/reprocess", response_model=SegmentResponse)
async def reprocess_segment(
    segment_id: UUID,
    data: str = Form(...),
    faceswap_image: UploadFile | None = File(None),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    try:
        body = SegmentReprocessRequest.model_validate_json(data)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid JSON in data field: {e}",
        )

    result = await db.execute(
        select(Segment)
        .join(Job, Segment.job_id == Job.id)
        .where(Segment.id == segment_id, Job.user_id == user.id)
    )
    segment = result.scalar_one_or_none()
    if segment is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Segment not found")
    if segment.status != SegmentStatus.COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Only completed segments can be reprocessed (current: '{segment.status}')",
        )

    job = await db.get(Job, segment.job_id)
    if job.status == JobStatus.ARCHIVED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot reprocess segments in an archived job",
        )

    # Upload faceswap source image if provided
    faceswap_uri = None
    if faceswap_image is not None:
        faceswap_uri = await upload_faceswap_image(
            faceswap_image, segment.job_id, key_suffix="faceswap_source_reprocess"
        )

    # Validate faceswap image is resolvable
    effective_image = faceswap_uri or body.faceswap_image
    if body.faceswap_enabled and not effective_image and body.faceswap_source_type != "start_frame":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Faceswap is enabled but no source image was provided",
        )

    # Update faceswap fields
    segment.faceswap_enabled = body.faceswap_enabled
    segment.faceswap_method = body.faceswap_method
    segment.faceswap_source_type = body.faceswap_source_type
    segment.faceswap_image = effective_image
    segment.faceswap_faces_order = body.faceswap_faces_order
    segment.faceswap_faces_index = body.faceswap_faces_index
    segment.faceswap_model = body.faceswap_model
    segment.faceswap_pixel_boost = body.faceswap_pixel_boost

    # Reset segment for faceswap-only reprocessing (preserve existing output)
    segment.reprocess_type = "faceswap"
    segment.status = SegmentStatus.PENDING
    segment.worker_id = None
    segment.worker_name = None
    segment.claimed_at = None
    segment.completed_at = None
    segment.error_message = None
    segment.progress_log = None

    # Reset job so daemon picks it up
    job.status = JobStatus.PENDING

    await db.commit()
    await db.refresh(segment)
    return segment


@router.post("/jobs/{job_id}/hologram", response_model=SegmentResponse)
async def make_hologram(
    job_id: UUID,
    body: HologramRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Create an AR hologram from a finalized job's stitched video.

    Reuses the reprocess queue: the job's lowest-index segment becomes the carrier
    (reprocess_type="ar_hologram"); the daemon mattes the job's final video into a packed
    color+alpha hologram + manifest + poster. The carrier's own output is preserved.
    """
    job = await db.get(Job, job_id)
    if job is None or job.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    if job.status != JobStatus.FINALIZED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Only finalized jobs can be turned into holograms (current: '{job.status}')",
        )

    holo_video = (
        await db.execute(
            select(Video)
            .where(Video.job_id == job.id, Video.status == VideoStatus.COMPLETED)
            .order_by(Video.created_at.desc())
        )
    ).scalars().first()
    if holo_video is None or not holo_video.output_path:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Job has no finalized video to source the hologram from",
        )

    # Resolve params: request override -> AppSetting global -> hardcoded default.
    key_setting = await db.get(AppSetting, "hologram_key_color")
    key_color = body.key_color or (key_setting.value if key_setting else None) or "0x00b140"
    height_setting = await db.get(AppSetting, "hologram_subject_height_m")
    subject_height = (
        body.subject_height_m
        if body.subject_height_m is not None
        else (float(height_setting.value) if height_setting else 1.70)
    )
    flavor_setting = await db.get(AppSetting, "hologram_flavor")
    flavor = body.flavor or (flavor_setting.value if flavor_setting else None) or "2d_matte"
    if flavor not in ("2d_matte", "2.5d_depth"):
        flavor = "2d_matte"
    depth_scale = body.depth_scale_m
    if depth_scale is not None:
        depth_scale = max(0.03, min(0.60, float(depth_scale)))

    # Dedicated hologram carrier at a sentinel index — a separate queue item, so the real
    # video segments and the job's FINALIZED status are left completely untouched. Reused
    # (reset) on re-make. The daemon claims it via the ar_hologram claim path; the console
    # hides it from the segment list.
    carrier = (
        await db.execute(
            select(Segment).where(
                Segment.job_id == job.id, Segment.index == HOLOGRAM_CARRIER_INDEX
            )
        )
    ).scalars().first()
    if carrier is None:
        carrier = Segment(
            job_id=job.id,
            index=HOLOGRAM_CARRIER_INDEX,
            prompt="[ar_hologram]",
            status=SegmentStatus.PENDING,
        )
        db.add(carrier)
    else:
        carrier.status = SegmentStatus.PENDING
        carrier.worker_id = None
        carrier.worker_name = None
        carrier.claimed_at = None
        carrier.completed_at = None
        carrier.error_message = None
        carrier.progress_log = None
        carrier.hologram_video_path = None
        carrier.hologram_manifest_path = None
        carrier.hologram_poster_path = None
    carrier.reprocess_type = "ar_hologram"
    carrier.hologram_key_color = key_color
    carrier.hologram_subject_height_m = subject_height
    carrier.hologram_flavor = flavor
    carrier.hologram_depth_scale_m = depth_scale

    await db.commit()
    await db.refresh(carrier)
    return carrier


@router.get("/segments/clips", response_model=list[SegmentClipResponse])
async def list_segment_clips(
    favorites_only: bool = Query(False),
    width: int = Query(None, description="Resolution filter (with height) for the smashcut res-lock"),
    height: int = Query(None),
    limit: int = Query(60, le=200),
    offset: int = Query(0, ge=0),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List completed real-generation segments as pickable clips for the Foundry pool / smashcut."""
    fav_rows = (
        await db.execute(
            select(Favorite.item_ref).where(
                Favorite.user_id == user.id, Favorite.item_type == "segment"
            )
        )
    ).scalars().all()
    fav_ids = set(fav_rows)

    where = [
        Job.user_id == user.id,
        Segment.status == SegmentStatus.COMPLETED,
        Segment.output_path.is_not(None),
        Segment.reprocess_type.is_(None),          # real generations only, not carriers/faceswap
        Segment.index < HOLOGRAM_CARRIER_INDEX,     # exclude sentinel carriers
    ]
    if width is not None and height is not None:
        where += [Job.width == width, Job.height == height]
    if favorites_only:
        if not fav_ids:
            return []
        where.append(Segment.id.in_([UUID(r) for r in fav_ids]))

    rows = (
        await db.execute(
            select(Segment, Job)
            .join(Job, Segment.job_id == Job.id)
            .where(*where)
            .order_by(Segment.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
    ).all()

    return [
        SegmentClipResponse(
            id=seg.id,
            job_id=job.id,
            job_name=job.name,
            index=seg.index,
            output_path=seg.output_path,
            thumbnail_path=seg.last_frame_path,
            width=job.width,
            height=job.height,
            fps=job.fps,
            duration_seconds=seg.duration_seconds,
            motion_magnitude=seg.motion_magnitude,
            favorite=str(seg.id) in fav_ids,
        )
        for seg, job in rows
    ]


SMASHCUT_SPEED_MIN = 0.25
SMASHCUT_SPEED_MAX = 4.0


def normalize_clip_speeds(clip_count: int, speeds: list[float] | None) -> list[float] | None:
    """Validate per-clip smashcut playback speeds, aligned 1:1 with the ordered clips.

    Collapses to None when nothing is actually retimed, which keeps the daemon on its fast
    stream-copy concat path — the field is a request for extra work, so an all-1.0 list must
    not read as one. Bounds match Segment.speed's, though the two are unrelated: that one is
    generation-time motion density, this one is playback rate on a finished clip.
    """
    if speeds is None:
        return None
    if len(speeds) != clip_count:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Expected {clip_count} clip speeds to match the picked clips, got {len(speeds)}",
        )
    for s in speeds:
        if not SMASHCUT_SPEED_MIN <= s <= SMASHCUT_SPEED_MAX:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Clip speed {s}x is outside {SMASHCUT_SPEED_MIN}x-{SMASHCUT_SPEED_MAX}x",
            )
    if all(s == 1.0 for s in speeds):
        return None
    return [float(s) for s in speeds]


@router.post("/smashcut", response_model=SegmentResponse)
async def create_smashcut(
    body: SmashcutRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Assemble hand-picked segment clips into one hard-cut montage.

    Creates a minimal container Job (no start image) hosting a smashcut carrier segment; the
    daemon concatenates the ordered clips (seamless butt-splice or dip-to-black) into the
    container's finalized Video. All clips must share resolution (enforced client-side + here).
    """
    if len(body.segment_ids) < 2:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Pick at least 2 clips")
    if body.transition not in ("seamless", "black"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid transition")
    clip_speeds = normalize_clip_speeds(len(body.segment_ids), body.clip_speeds)

    # Resolve the picked segments (must be the user's, completed, with output). Preserve order.
    rows = (
        await db.execute(
            select(Segment, Job)
            .join(Job, Segment.job_id == Job.id)
            .where(Segment.id.in_(body.segment_ids), Job.user_id == user.id)
        )
    ).all()
    by_id = {seg.id: (seg, job) for seg, job in rows}
    ordered = [by_id.get(sid) for sid in body.segment_ids]
    if any(item is None for item in ordered):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="One or more clips not found")
    clip_paths = []
    for seg, _job in ordered:
        if seg.status != SegmentStatus.COMPLETED or not seg.output_path:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="A clip is not a completed video")
        clip_paths.append(seg.output_path)

    # Resolution lock — all clips must match (first clip sets the montage resolution).
    first_job = ordered[0][1]
    w, h, fps = first_job.width, first_job.height, first_job.fps
    if any(job.width != w or job.height != h for _seg, job in ordered):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="All clips must share resolution")

    # Minimal container job (no start image). FINALIZING so the GPU claim ignores it; the CPU
    # reprocess track claims the carrier regardless of job status, then upload sets FINALIZED.
    container = Job(
        user_id=user.id,
        name=body.name.strip() or "Smashcut",
        width=w, height=h, fps=fps,
        seed=random.randint(0, 2**63 - 1),
        status=JobStatus.FINALIZING,
        tags="smashcut",
    )
    db.add(container)
    await db.flush()

    carrier = Segment(
        job_id=container.id,
        index=SMASHCUT_CARRIER_INDEX,
        prompt="[smashcut]",
        status=SegmentStatus.PENDING,
        reprocess_type="smashcut_concat",
        smashcut_clip_paths=clip_paths,
        smashcut_transition=body.transition,
        smashcut_clip_speeds=clip_speeds,
    )
    db.add(carrier)
    await db.commit()
    await db.refresh(carrier)
    return carrier


@router.get("/segments/{segment_id}/hologram")
async def get_hologram(
    segment_id: UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return a segment's hologram artifact S3 paths (for the WebXR AR player)."""
    result = await db.execute(
        select(Segment)
        .join(Job, Segment.job_id == Job.id)
        .where(Segment.id == segment_id, Job.user_id == user.id)
    )
    segment = result.scalar_one_or_none()
    if segment is None or not segment.hologram_video_path:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Hologram not found")
    return {
        "flavor": segment.hologram_flavor or "2d_matte",
        "video_path": segment.hologram_video_path,
        "manifest_path": segment.hologram_manifest_path,
        "poster_path": segment.hologram_poster_path,
        # Cache-buster for the player: hologram artifacts live at fixed S3 keys and the /files
        # redirect is cached for hours, so a remake would otherwise replay the previous flavor.
        "version": segment.completed_at.isoformat() if segment.completed_at else None,
    }


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
