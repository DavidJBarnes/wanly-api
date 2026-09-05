import asyncio
import base64
import json
import asyncio
import logging
import math
import random
import re
import subprocess
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Query, UploadFile, status
from sqlalchemy import and_, func, or_, select, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.auth import get_current_user, verify_api_key, verify_api_key_or_bearer
from app import s3
from app.database import get_db
from app.routes.captions import caption_image_bytes
from app.seeds import new_seed
from app.enums import JobStatus, SegmentStatus, VideoStatus
from app.ltx_stack import LTX_STACK
from app.model_requirements import CHECKPOINT, canonical
from app.models import AppSetting, Job, LtxCharacter, Segment, User, Video, Wildcard, Worker
from app.s3 import delete_object, download_file, move_object, parse_s3_uri
from app.schemas.segments import (
    SegmentPromptUpdate,
    FramePreview,
    FramePreviewResponse,
    HologramRequest,
    RerollRequest,
    SegmentClaimResponse,
    SegmentClipResponse,
    SegmentCreate,
    SegmentResponse,
    SegmentStatusUpdate,
    SegmentTrimUpdate,
    SmashcutRequest,
    WorkerSegmentResponse,
)
from app.models import Favorite
from app.stitch import stitch_video

logger = logging.getLogger(__name__)

# A worker heartbeats every 30s (daemon HEARTBEAT_INTERVAL), so 5 minutes is ~10 missed beats -
# comfortably past a transient network blip. Renders themselves have no upper bound (60fps jobs
# run 40+ minutes); a claim is held for as long as its worker keeps heartbeating.
STALE_HEARTBEAT_MINUTES = 5
# How long an idle worker may hold a claim it has written no progress against before the
# claim is treated as orphaned. Two minutes is generous: the daemon posts its first progress
# line within seconds of receiving a segment. See wanly-api#242.
ORPHANED_CLAIM_MINUTES = 2

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


SCENE_PLACEHOLDER = "<SCENE>"


async def _resolve_scene(db: AsyncSession, prompt: str, image_uri: str | None,
                         *, final: bool = False) -> str:
    """Fill a pose's <SCENE> with a description of the frame this segment starts from.

    Recipes are start-frame-agnostic by design — one pose serves every character — so their
    static half is a generic guess about an image they have never seen. A validated pose
    reads "a woman kneeling in front of a nude man" while the actual start frame is a
    clothed woman sitting on a sofa. <SCENE> lets the pose defer that half to the frame.

    ORDER: this runs AFTER _resolve_wildcards, the opposite of _resolve_trigger, and the
    reason is the caption text. A caption is model output that may contain bracketed tokens,
    and running the wildcard resolver over it afterwards would expand them. Going last means
    the caption is never re-scanned.

    That ordering removes the protection _resolve_trigger gets for free — a wildcard named
    SCENE would be substituted before we ever look — so SCENE is RESERVED in the wildcard
    routes. For <TRIGGER> the reservation is belt-and-braces; here it is the only guard.

    FAILURE INVERTS FROM <TRIGGER>. _resolve_trigger deliberately leaves the literal
    placeholder when it cannot resolve, because silently dropping the token that anchors the
    character LoRA is worse and harder to notice. Here the reasoning flips: a literal
    "<SCENE>" is garbage tokens fed to the text encoder, while dropping it leaves a valid if
    generic prompt. So it is dropped, and the drop is logged.

    `final` is which of the two resolution points this is. At SUBMIT (final=False) a missing
    image means "not yet" — a continuation is routinely created before the segment it
    follows has rendered — so the placeholder survives for the claim to resolve. At CLAIM
    (final=True) the start image is as known as it will ever be, so an unresolved
    placeholder is dropped rather than shipped to the encoder.
    """
    if SCENE_PLACEHOLDER not in prompt:
        return prompt

    if not image_uri:
        if final:
            # Last responsible moment and still no image: a text-to-video segment, or a
            # continuation whose predecessor produced no last frame. Nothing to describe.
            logger.info("Dropping %s: no start image to describe", SCENE_PLACEHOLDER)
            return _drop_scene(prompt)
        # A continuation submitted before its predecessor has rendered. The frame it will
        # start from does not exist yet, so the placeholder is LEFT IN PLACE and resolved at
        # claim time, when the previous segment's last frame is known.
        #
        # Dropping it here instead would be unrecoverable: the placeholder would be gone
        # from the stored prompt and no later step could tell that a description was ever
        # wanted. Continuations are the case this feature helps most — they condition on a
        # generated frame nobody has ever described — so losing them silently is the worst
        # available outcome.
        logger.info("Deferring %s: start image not known until claim", SCENE_PLACEHOLDER)
        return prompt

    try:
        image = await asyncio.to_thread(s3.download_bytes, image_uri)
        caption, _ = await caption_image_bytes(db, image)
    except Exception as e:
        # Never fatal. A captioner that is down must not stop a render — the pose still has
        # its arc, and a generic scene is what every pose used before this existed.
        logger.warning("Dropping %s: could not describe %s (%s)",
                       SCENE_PLACEHOLDER, image_uri, e)
        return _drop_scene(prompt)

    logger.info("Resolved %s from %s: %r", SCENE_PLACEHOLDER, image_uri, caption[:80])
    return prompt.replace(SCENE_PLACEHOLDER, caption)


def _drop_scene(prompt: str) -> str:
    """Remove the placeholder and the comma-space that would be left dangling beside it.

    "<TRIGGER>, <SCENE>, she grips..." must not become "trigger, , she grips...", which
    reaches the encoder as a stray empty clause.
    """
    out = prompt.replace(f", {SCENE_PLACEHOLDER},", ",")
    out = out.replace(f"{SCENE_PLACEHOLDER}, ", "").replace(f", {SCENE_PLACEHOLDER}", "")
    return " ".join(out.replace(SCENE_PLACEHOLDER, "").split())


async def _resolve_trigger(db: AsyncSession, prompt: str, ltx_recipe: dict | None) -> str:
    """Fill a pose's <TRIGGER> with the character's trigger word.

    Runs BEFORE _resolve_wildcards, and that order is the safeguard: <TRIGGER> shares syntax
    with wildcards, so a Wildcard named TRIGGER would otherwise substitute a random option and
    the render would quietly name the wrong character. Doing it first means the resolver never
    sees the placeholder. (The name is also reserved in the wildcard routes, so both ends are
    covered.)

    The console normally substitutes at creation time; this catches a prompt that reaches the
    API still carrying the placeholder — an edited prompt, or any caller that is not the
    console.
    """
    if "<TRIGGER>" not in prompt:
        return prompt
    name = (ltx_recipe or {}).get("character")
    if not name:
        # Leave it. Rendering the literal text "<TRIGGER>" is bad, but silently dropping the
        # token that anchors the character LoRA is worse and much harder to notice.
        return prompt
    row = (await db.execute(
        select(LtxCharacter).where(LtxCharacter.name == name)
    )).scalar_one_or_none()
    return prompt.replace("<TRIGGER>", row.trigger) if row else prompt


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

    prompt = await _resolve_trigger(db, body.prompt, body.ltx_recipe)
    resolved_prompt, prompt_template = await _resolve_wildcards(db, prompt)
    # After wildcards, deliberately — see _resolve_scene. The console resolves this itself
    # for segment 0, where a person can review the caption first; this catches a prompt that
    # arrives still carrying the placeholder, which is every continuation and any caller
    # that is not the console. Same convention as <TRIGGER>.
    resolved_prompt = await _resolve_scene(db, resolved_prompt, body.start_image)

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
        negative_prompt=negative_prompt,
        ltx_recipe=body.ltx_recipe,
        auto_finalize=body.auto_finalize,
    )
    db.add(segment)

    job.status = JobStatus.PENDING

    await db.commit()
    await db.refresh(segment)
    return segment


def _model_gate(worker: Worker | None) -> tuple:
    """SQL that hides segments this worker cannot render. console#422.

    A pod carries sulphur and nothing else. Before this, it claimed a 10Eros pose anyway,
    uploaded the start image, and ComfyUI rejected the graph across three loaders — after the
    claim, so the segment failed and the job stalled beside a 3090 that had the file.

    IN THE QUERY, not after it. Filtering the selected row in Python would either re-pick the
    same unrunnable segment on every poll — head-of-line blocking, with the whole queue behind
    it — or need a second SELECT that breaks the skip-locked concurrency the claim depends on.

    Three things deliberately do NOT gate:

      * a NULL ltx_recipe. Not a recipe render: a WAN segment, a free-form LTX one, or a CPU
        reprocess carrier. It declares no models, so it requires none. Conservative on
        purpose — this must not withhold work that flows today.
      * a worker that has never reported its checkpoints (NULL, not []). An older daemon
        would otherwise starve on upgrade day, and a fleet claiming nothing looks exactly
        like an empty queue.
      * a kind the worker says it can fetch. It already downloads LoRAs at claim time; when
        it learns to fetch checkpoints (console#423) this stops gating on its own.

    Inventory comes from the last heartbeat rather than the claim request: the poll is
    frequent and the inventory is not. The staleness window is one heartbeat, and a claim
    made against a stale one degrades to exactly the old behaviour — a loud engine failure —
    rather than to something worse.
    """
    if worker is None or worker.checkpoints is None:
        return ()
    if CHECKPOINT in (worker.fetchable_kinds or []):
        return ()

    names = {canonical(n) for n in worker.checkpoints if isinstance(n, str) and n.strip()}
    # Both spellings. Recipes store a bare stem and the daemon strips the extension before
    # reporting, so these agree today; if that convention ever drifts the gate does not fail
    # loudly, it silently matches nothing and the worker claims no work at all.
    names |= {f"{n}.safetensors" for n in names}

    checkpoint = func.coalesce(
        func.nullif(Segment.ltx_recipe["checkpoint"].astext, ""),
        canonical(LTX_STACK["checkpoint"]),
    )
    return (or_(Segment.ltx_recipe.is_(None), checkpoint.in_(names)),)


@router.get("/segments/next", dependencies=[Depends(verify_api_key)])
async def claim_next_segment(
    worker_id: UUID = Query(...),
    worker_name: str = Query(None),
    kind: str = Query(None, description="'gpu' (exclude holograms), 'hologram' (only holograms), or None (any)"),
    db: AsyncSession = Depends(get_db),
):
    # Reclaim work orphaned by a dead worker.
    #
    # The heartbeat is the authority on dead-vs-alive: workers beat every 30s, so a dead one is
    # detectable in ~5 minutes, and a worker that IS heartbeating keeps its claim no matter how
    # long the render runs. The age rule applies only when there is no heartbeat to consult —
    # the worker row is gone (RunPod pods delete theirs on drain; a claim can outlive its row).
    #
    # The age rule must never apply to a live worker. When it did, every render longer than 30
    # minutes was stolen mid-flight by the other GPU's poll, both GPUs converged on rendering
    # the same segment, and fleet throughput silently dropped to one GPU (2026-08-15).
    now = datetime.now(timezone.utc)
    age_cutoff = now - timedelta(minutes=30)
    heartbeat_cutoff = now - timedelta(minutes=STALE_HEARTBEAT_MINUTES)
    orphan_cutoff = now - timedelta(minutes=ORPHANED_CLAIM_MINUTES)

    stale_result = await db.execute(
        select(Segment, Worker.last_heartbeat)
        .outerjoin(Worker, Worker.id == Segment.worker_id)
        .where(
            Segment.status.in_([SegmentStatus.CLAIMED, SegmentStatus.PROCESSING]),
            Segment.claimed_at.is_not(None),
            or_(
                Worker.last_heartbeat < heartbeat_cutoff,
                # last_heartbeat is NOT NULL on the workers table, so "no heartbeat to
                # consult" is exactly "the worker row is gone" (outerjoin came up empty).
                and_(Segment.claimed_at < age_cutoff, Worker.id.is_(None)),
                # A claim whose RESPONSE was lost. GET /segments/next mutates: the segment is
                # marked and assigned before the answer is sent, so a transport failure on the
                # way back leaves it owned by a worker that never learned of it. That worker
                # stays healthy and heartbeating, so neither rule above ever fires and the
                # segment is stuck forever with the queue stopped beside an idle GPU
                # (wanly-api#242, hit in production 2026-09-03).
                #
                # THREE conditions, all required, because reclaiming from a LIVE worker is
                # how two GPUs once converged on one segment (see the warning above):
                #
                #   1. the worker says it is idle. The daemon sets online-busy immediately on
                #      receiving a claim, BEFORE rendering — so a worker that got the answer
                #      is never idle here.
                #   2. no progress has been written. A running segment logs "[1/6] ..." within
                #      seconds. This is evidence from the SEGMENT, and it is what makes the
                #      rule safe: the daemon's status push can fail, and the heartbeat only
                #      re-pushes on a CHANGE, so status alone can lie. An empty progress log
                #      cannot.
                #   3. it has sat like that past the grace period, which is far longer than
                #      the round trip between claiming and the first progress line.
                and_(
                    Worker.status == "online-idle",
                    # Heartbeating RIGHT NOW. This is what rules out a network partition:
                    # the worry with the two checks below is a worker that is really
                    # rendering but whose status push and progress writes both failed. If it
                    # is beating, the API is reachable from that worker, so those writes had
                    # no reason to fail — and a worker that has genuinely gone quiet is the
                    # heartbeat rule's business, not this one's.
                    Worker.last_heartbeat >= heartbeat_cutoff,
                    Segment.claimed_at < orphan_cutoff,
                    or_(Segment.progress_log.is_(None), Segment.progress_log == ""),
                ),
            ),
        )
    )
    for stale, last_heartbeat in stale_result.all():
        # Naming which rule fired matters: "the worker died" and "the worker is fine but
        # never received the answer" have completely different fixes, and the log line is
        # where that is decided at 2am.
        if last_heartbeat is None:
            reason = "worker row gone and claimed over 30m ago"
        elif last_heartbeat < heartbeat_cutoff:
            reason = "worker heartbeat stale"
        else:
            reason = ("claim response lost — worker is idle with no progress logged "
                      "(wanly-api#242)")
        logger.warning(
            "Reclaiming segment %s (status=%s, claimed_at=%s, worker=%s, last_heartbeat=%s): %s",
            stale.id, stale.status, stale.claimed_at, stale.worker_name, last_heartbeat, reason,
        )
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
    #
    # Every branch excludes discarded segments. Status alone does not: a discarded segment keeps
    # the status it was discarded in, and retrying a failed one puts it back to PENDING — at
    # which point the worker regenerates a take the operator has already archived, silently
    # spending GPU time on a clip nobody asked for.
    if kind == "hologram":
        where = (
            Segment.status == SegmentStatus.PENDING,
            Segment.discarded.is_(False),
            Segment.reprocess_type.in_(_CPU_REPROCESS_TYPES),
        )
    elif kind == "gpu":
        where = (
            Segment.status == SegmentStatus.PENDING,
            Segment.discarded.is_(False),
            Job.status.in_([JobStatus.PENDING, JobStatus.PROCESSING]),
            # real generations (reprocess_type NULL); exclude the CPU reprocess carriers.
            # NULL NOT IN (...) is NULL (excludes), so OR the NULL case in explicitly.
            or_(
                Segment.reprocess_type.is_(None),
                Segment.reprocess_type.not_in(_CPU_REPROCESS_TYPES),
            ),
        )
    else:
        where = (
            Segment.status == SegmentStatus.PENDING,
            Segment.discarded.is_(False),
            or_(
                Job.status.in_([JobStatus.PENDING, JobStatus.PROCESSING]),
                Segment.reprocess_type.in_(_CPU_REPROCESS_TYPES),
            ),
        )
    # Read once, used twice: the gate needs this worker's inventory before the query, and the
    # GPU name is snapshotted off the same row after it.
    claiming_worker = await db.get(Worker, worker_id)

    result = await db.execute(
        select(Segment)
        .join(Job, Segment.job_id == Job.id)
        .where(*where, *_model_gate(claiming_worker))
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

    # Snapshot the GPU now, not at read time. Worker rows are deleted on deregister — every
    # RunPod pod takes its row with it when it drains — so a later join would silently lose the
    # hardware for every segment a since-terminated pod ran.
    if claiming_worker and claiming_worker.gpu_stats:
        segment.gpu_name = claiming_worker.gpu_stats.get("gpu_name")

    job = await db.get(Job, segment.job_id)
    if job.status == JobStatus.PENDING:
        job.status = JobStatus.PROCESSING

    # Resolve start_image
    resolved_start_image = segment.start_image
    previous_segment = None
    reference_frames = []
    if resolved_start_image is None:
        if segment.index == 0:
            resolved_start_image = job.starting_image
        else:
            # The LIVE segment at the previous index. Index alone stopped identifying one row
            # when re-rolling arrived: an archived take keeps its index, so a job rolled six
            # times has seven rows at index 0 and this raised MultipleResultsFound — a 500 on
            # every claim, which stops the queue dead rather than degrading.
            #
            # Live is also the right answer on its own terms: this resolves the frame the next
            # segment continues from, and a discarded take is not what anything continues from.
            prev_result = await db.execute(
                select(Segment)
                .where(
                    Segment.job_id == job.id,
                    Segment.index == segment.index - 1,
                    Segment.discarded.is_(False),
                )
            )
            prev_segment = prev_result.scalar_one_or_none()
            if prev_segment is not None:
                resolved_start_image = prev_segment.last_frame_path
                previous_segment = prev_segment
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

    # <SCENE> for a continuation. The start frame is a GENERATED image that exists only now
    # — the previous segment produced it — so this is the first moment it can be described,
    # and it is the case the feature exists for: nobody has ever written a description of
    # that frame, and the recipe's own scene wording was authored before it existed.
    #
    # On the claim path deliberately, despite this being polled. A caption is 4.5s cold and
    # 1.2s warm (measured), against a segment that then renders for ten minutes, and it only
    # runs for a prompt that actually carries the placeholder. An earlier draft of this work
    # avoided the claim path on the strength of a 60-90s estimate that turned out to be
    # wall-clock on a multi-prompt script, and was wrong.
    if SCENE_PLACEHOLDER in (segment.prompt or ""):
        resolved = await _resolve_scene(db, segment.prompt, resolved_start_image, final=True)
        if resolved != segment.prompt:
            # Persisted, not just returned: the segment must record what it actually ran, so
            # a retry reproduces it and a rated result can be traced to its real prompt.
            segment.prompt = resolved

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
        reference_frames=reference_frames if reference_frames else None,
        negative_prompt=negative_prompt,
        # Passed through verbatim. The engine must never look a recipe up for itself: an
        # engine that cannot look one up cannot look up a STALE one, which is the failure
        # this shape exists to make impossible (see wanly-api#207).
        ltx_recipe=segment.ltx_recipe,
        reprocess_type=segment.reprocess_type,
        output_path=segment.output_path,
        width=job.width,
        height=job.height,
        fps=job.fps,
        # THE SEED IS LOCKED ACROSS A CHAIN. Next Segment must keep the seed -- a
        # continuation should look like the same shot carrying on, and the seed drives how a
        # take looks more than anything else. Re-roll must CHANGE it, which is the opposite
        # requirement and is served elsewhere: _roll_new_take sets a new job.seed, so the
        # fresh take and every continuation after it inherit that instead.
        #
        # job.seed is therefore always the live chain's seed. A segment's own seed is only
        # ever set on a DISCARDED take, stamped at re-roll time to record what it actually
        # ran on, and a discarded take is never claimed.
        #
        # It used to be job.seed + index, which decorrelated later segments "so they don't
        # repeat the same motion pattern". That was WAN reasoning: WAN drifted, and varying the
        # seed spread the drift around. Under LTX a continuation is meant to look like the same
        # shot continuing, and the seed is the single biggest determinant of what a take looks
        # like -- expression especially is seed-driven far more than it is LoRA-driven.
        seed=segment.seed if segment.seed is not None else job.seed,
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


@router.patch("/segments/{segment_id}/prompt", response_model=SegmentResponse)
async def update_segment_prompt(
    segment_id: UUID,
    body: SegmentPromptUpdate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Change a queued segment's prompt, or re-take it from its preset.

    Only while the segment has not been claimed. Once a worker holds it the prompt has already
    been sent to ComfyUI, so editing would change the record without changing the output -- the
    worst outcome, because the row would then describe something the video is not.

    Wildcards are resolved here exactly as they are at creation. Skipping that would make an
    edited prompt behave differently from an identical one typed into the create dialog, and
    `<face>` would reach the model as literal text.
    """
    result = await db.execute(
        select(Segment)
        .join(Job, Segment.job_id == Job.id)
        .where(Segment.id == segment_id, Job.user_id == user.id)
    )
    segment = result.scalar_one_or_none()
    if segment is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Segment not found")

    if segment.status != SegmentStatus.PENDING or segment.worker_id is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Segment is {segment.status} and cannot be edited. A prompt can only be changed "
                "before a worker claims it; after that the change would not reach the video."
            ),
        )

    source = body.prompt

    resolved, template = await _resolve_wildcards(db, source)
    segment.prompt = resolved
    segment.prompt_template = template
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
    # Retrying an archived take asks for a clip that was deliberately thrown away. Claiming
    # already refuses to hand one out, so without this the retry would look like it worked and
    # then sit PENDING forever.
    if segment.discarded:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This take was discarded. Restore it before retrying, or re-roll instead.",
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
        Segment.reprocess_type.is_(None),          # real generations only, not carriers
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


def _recipe_for_take(old: Segment, prompt: str | None) -> dict | None:
    """The outgoing take's recipe, marked if this roll changed the prompt.

    A re-roll's premise is that the seed is the ONLY difference, which is what makes two takes
    comparable. Changing the prompt deliberately breaks that — and six takes later nothing
    distinguishes a prompt-changed pair from a seed-only one, so a judgement made across them
    is quietly worthless.

    `edited` already carries exactly this meaning: the console writes it at creation to record
    which of a pose's defaults the user overrode. A copy, never a mutation — the archived take
    and its replacement share the dict otherwise, and marking one would rewrite the record of
    the other.
    """
    recipe = old.ltx_recipe
    if recipe is None or prompt is None or prompt == old.prompt:
        return recipe
    return {**recipe, "edited": sorted({*(recipe.get("edited") or []), "prompt"})}


def _roll_new_take(job: Job, old: Segment, *, prompt: str | None = None,
                   prompt_template: str | None = None) -> Segment:
    """Archive `old` and build its replacement.

    Was shared by a user-initiated re-roll and an automatic rule-driven one; the rule-driven
    path is gone with the metrics it judged (#151), so this now serves the button only.

    Stamp the outgoing take with the seed it actually ran on, if it was relying on the
    derivation. NULL means "derive from the job", which is unambiguous while a job has one take
    and stops being so the moment it has several: the archive exists to answer "which seed gave
    me that one", and an archived clip whose seed has to be recomputed from context is a worse
    answer than a recorded number. Same value the worker used -- this records it, it does not
    change it.

    The job's seed then becomes the new take's seed, and the new take derives from it like any
    other segment. This is only safe because the outgoing take was just stamped: the seed being
    replaced is still recorded, on the row it actually produced. See #199 for the full model.

    ANY INDEX, not only 0 (console#424). The replacement takes the index it replaces, and only
    the LAST live segment may be rolled — a segment with a successor is the frame that
    successor continues from. The predecessors are untouched, so the new take starts from the
    same frame the archived one did, which is what keeps the pair comparable.

    An optional `prompt` is the second half of that ticket: "that take was close, let me nudge
    the wording". It arrives already resolved, because resolution needs the database and this
    does not touch it.
    """
    if old.seed is None:
        # Exactly what the worker ran: the claim hands out job.seed. This said
        # `(job.seed + old.index) % (2**63 - 1)`, the derivation the claim used before the
        # seed was locked across a chain. It survived because re-roll only ever acts on
        # index 0, where the two agree -- so it would have recorded a seed the take never
        # ran on the first time anyone re-rolled a continuation.
        old.seed = job.seed
    old.discarded = True

    # Every OTHER live take deriving its seed from the job has to be stamped too, before the
    # job's seed moves out from under it. Rolling index 0 never needed this — there was
    # nothing else live. Rolling a continuation does: segments 0..n-1 hold seed NULL, which
    # means "ask the job", and the job is about to answer differently. They would not just be
    # mislabelled; retrying one re-claims it, the claim hands out job.seed, and it would
    # render a different take from the one it is retrying.
    for sibling in job.segments:
        if sibling is not old and not sibling.discarded and sibling.seed is None:
            sibling.seed = job.seed

    job.seed = new_seed()

    fresh = Segment(
        job_id=job.id,
        # The index it replaces. Both rows hold it at once — the unique index is partial over
        # live rows — which is what lets an archived take keep its place in the chain.
        index=old.index,
        # NULL: derive from the job like every other segment. The job seed WAS just set to this
        # take's seed, so a second copy on the segment would be the same number in two places,
        # free to drift.
        seed=None,
        prompt=old.prompt if prompt is None else prompt,
        prompt_template=old.prompt_template if prompt is None else prompt_template,
        duration_seconds=old.duration_seconds,
        speed=old.speed,
        start_image=old.start_image,
        negative_prompt=old.negative_prompt,
        # The recipe is what the take IS. Without it a re-rolled LTX segment renders free-form
        # — no character LoRA, no trigger, no per-stage strengths — and comes back looking like
        # a different shot, which is the opposite of a re-roll's entire purpose.
        ltx_recipe=_recipe_for_take(old, prompt),
        auto_finalize=old.auto_finalize,
    )

    # Back to pending or nothing claims it. The job may well be 'awaiting' (segment 0 finished
    # and it is waiting on a decision) or 'failed'.
    job.status = JobStatus.PENDING
    return fresh


_REROLL_REFUSED_STATUSES = (JobStatus.FINALIZED, JobStatus.FINALIZING, JobStatus.ARCHIVED)


async def _reroll(db: AsyncSession, job: Job, old: Segment,
                  prompt: str | None) -> Segment:
    """Archive `old` and queue a fresh take of it. Shared by both routes.

    This is the "show me another take" button. Everything that defines the shot — LoRAs,
    recipe, start image, duration — is copied verbatim so that the seed is the only thing
    that differs and the two takes are actually comparable. A wildcard prompt is copied
    already-resolved for the same reason: re-rolling the wildcards too would change two
    variables at once and make the comparison worth nothing.

    Unless the caller asks for a different prompt, which is the point of console#424 and is
    an explicit choice rather than a side effect. The recipe records that it happened.

    ONLY THE LAST LIVE SEGMENT. A segment with a live successor is the frame that successor
    continues from; replacing it would leave every one of them continuing from a frame that
    no longer exists. That was expressed as "index must be 0" while re-roll only served
    single-segment jobs — it is the same protection, stated generally.

    The old take is discarded, not deleted. It keeps its video under its own seed, which is
    why segments have a seed column at all. Rolling repeatedly is the point, so an archive
    that relabelled every previous take with the newest seed would destroy exactly the record
    it exists to accumulate.

    Trim and transition are deliberately NOT copied: they describe the take being archived,
    not the one about to be generated.
    """
    if job.status in _REROLL_REFUSED_STATUSES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Job is '{job.status}' and cannot be re-rolled. Its stitched output "
                "describes the takes that were live when it was built; archiving one behind "
                "that video would make the record wrong."
            ),
        )

    live = [s for s in job.segments if not s.discarded]
    last = max(live, key=lambda s: s.index, default=None)
    if old.discarded or last is None or old.id != last.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Only the job's current segment can be re-rolled (that is index "
                f"{last.index if last else '-'}, this is {old.index}). A later segment "
                "continues from this one's last frame, so replacing it would orphan them."
            ),
        )

    if old.status not in (SegmentStatus.FAILED, SegmentStatus.COMPLETED):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Segment {old.index} is '{old.status}'. Cancel it before re-rolling — "
                "otherwise the worker finishes generating a take that has already been "
                "archived."
            ),
        )

    resolved, template = None, None
    if prompt is not None:
        # Exactly the creation path's order. <TRIGGER> first, because it shares syntax with
        # wildcards and a Wildcard named TRIGGER would otherwise substitute a random option
        # and the render would quietly name the wrong character.
        #
        # <SCENE> is deliberately left alone: for a continuation it describes a frame that
        # does not exist yet, and the claim resolves it then — the same way a continuation
        # created through the normal path is handled.
        triggered = await _resolve_trigger(db, prompt, old.ltx_recipe)
        resolved, template = await _resolve_wildcards(db, triggered)

    fresh = _roll_new_take(job, old, prompt=resolved, prompt_template=template)
    db.add(fresh)
    await db.commit()
    await db.refresh(fresh)
    return fresh


@router.post("/segments/{segment_id}/reroll", response_model=SegmentResponse)
async def reroll_segment(
    segment_id: UUID,
    body: RerollRequest | None = None,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Another take of this segment, optionally with a different prompt (console#424).

    Addressed by SEGMENT rather than by job. "The job's one segment" was unambiguous while
    re-roll only served index 0; once any index can be the target, the caller has to be able
    to say which one it meant — and the API has to be able to refuse when that is no longer
    the current segment, rather than rolling something the user was not looking at.
    """
    result = await db.execute(
        select(Segment)
        .join(Job, Segment.job_id == Job.id)
        .where(Segment.id == segment_id, Job.user_id == user.id)
    )
    segment = result.scalar_one_or_none()
    if segment is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Segment not found")

    job = (await db.execute(
        select(Job).where(Job.id == segment.job_id).options(selectinload(Job.segments))
    )).scalar_one()

    return await _reroll(db, job, segment, body.prompt if body else None)


@router.post("/jobs/{job_id}/reroll", response_model=SegmentResponse)
async def reroll_current_segment(
    job_id: UUID,
    body: RerollRequest | None = None,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Re-roll whatever the job's current segment is.

    Superseded by POST /segments/{id}/reroll and kept because a browser holding the previous
    console still calls it — dropping it would break the button for as long as a stale tab
    lives. For the single-segment job that console offers it on, the two are identical.
    """
    result = await db.execute(
        select(Job)
        .where(Job.id == job_id, Job.user_id == user.id)
        .options(selectinload(Job.segments))
    )
    job = result.scalar_one_or_none()
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")

    live = [s for s in job.segments if not s.discarded]
    if not live:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Job has no live segment to re-roll.")

    return await _reroll(db, job, max(live, key=lambda s: s.index),
                         body.prompt if body else None)


@router.post("/segments/{segment_id}/discard", response_model=SegmentResponse)
async def discard_segment(
    segment_id: UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Take a segment out of the video without losing the take itself.

    Deleting destroys the row, and with it the record that this seed produced this clip. A bad
    take is frequently the most informative one, and destroying it in order to get it out of
    the cut is exactly backwards.

    The row keeps its index, so the record reads as "the discarded version of segment 2" and a
    regenerated segment can take the same position. The unique constraint is partial for that
    reason -- see migration 063.

    Output files are deliberately NOT removed. The clip IS the record -- a discarded take that
    cannot be rewatched is not archived, it is deleted with extra steps.
    """
    result = await db.execute(
        select(Segment)
        .join(Job, Segment.job_id == Job.id)
        .where(Segment.id == segment_id, Job.user_id == user.id)
    )
    segment = result.scalar_one_or_none()
    if segment is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Segment not found")
    if segment.discarded:
        return segment
    if segment.status not in (SegmentStatus.FAILED, SegmentStatus.COMPLETED):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Only failed or completed segments can be discarded (current: "
                f"'{segment.status}'). Cancel a running segment first."
            ),
        )

    segment.discarded = True
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
    for path in [segment.output_path, segment.last_frame_path]:
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
