"""Character-LoRA training as claimable work (wanly-api#274, wanly-console#453).

Training a LoRA has been a laptop-driven ssh pipeline. This makes it a row the console creates
and a trainer claims, which is how every other long remote job in this system works.

PULL, NOT PUSH, and that is the whole design decision. The alternative -- the console or the API
calling a trainer over HTTP -- would need a trainer URL in config, a retry policy, and its own
answer to "the trainer restarted mid-run". Claiming gets orphan reclaim, the heartbeat/offline
sweep and queue-health for free, because they already exist for segments and key on the same
columns.

WHAT IS DELIBERATELY NOT HERE: any knowledge of how a LoRA is trained. The recipe -- rank, LR,
steps, the preset -- is snapshotted into `config` when the job is created and handed to the
trainer verbatim. This module could not tell you what rank 32 means, and should not be able to.
"""
import asyncio
import logging
import uuid
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app import s3
from app.auth import get_current_user, verify_api_key, verify_api_key_or_bearer
from app.config import settings
from app.database import get_db
from app.enums import TRAINING_TERMINAL, TrainingStatus, WorkerKind
from app.models import Dataset, LtxCharacter, TrainingJob, User, Worker
from app.schemas.training import (
    MIN_DATASET_IMAGES, TrainingClaimResponse, TrainingCreate, TrainingProgress,
    TrainingResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()

#: Mirrors the segment claim's grace period: a claim held by a worker that says it is idle and
#: has written no progress is presumed lost after this. Longer than a segment's because a
#: trainer's first phase (staging a dataset, caching latents) is legitimately quiet for minutes.
ORPHANED_TRAINING_MINUTES = 20
#: A worker not heard from in this long is dead, whatever it last said.
STALE_HEARTBEAT_MINUTES = 5

#: What the recipe is, at the moment a job is created. Snapshotted rather than referenced so a
#: change here cannot retroactively alter what a queued job will do. The trainer owns the real
#: implementation; these are the values it is told to use.
RECIPE_DEFAULTS = {
    "network_dim": 32,
    "network_alpha": 32,
    "learning_rate": 1e-4,
    "lora_target_preset": "video_sa_ca_ff",
    "num_repeats": 10,
    "seed": 42,
}


def _default_lora_name(character: str) -> str:
    """A safe filename stem, as a STARTING POINT for the user to correct.

    Stripping is the honest default -- it never invents a letter -- but it is not always the
    right answer, which is why the field exists.
    """
    return "".join(c for c in character if c.isalnum() or c in "._-") or "lora"


def _live_states() -> list[str]:
    return [TrainingStatus.CLAIMED, TrainingStatus.RUNNING]


@router.post("/training", response_model=TrainingResponse, status_code=201)
async def create_training_job(
    body: TrainingCreate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Queue a training run. The console's entry point."""
    # A dataset is the normal path; a raw list stays supported so a CLI or a curl can train
    # without creating one first.
    images = list(body.dataset_images)
    if body.dataset_id:
        ds = await db.get(Dataset, body.dataset_id)
        if not ds:
            raise HTTPException(status_code=404, detail="Dataset not found")
        images = list(ds.images)
    # Checked here rather than on the schema, because it applies to whichever of the two was
    # given -- a schema minimum on dataset_images would reject every dataset_id request.
    if len(images) < MIN_DATASET_IMAGES:
        raise HTTPException(
            status_code=422,
            detail=f"{len(images)} images — at least {MIN_DATASET_IMAGES} are needed")
    if len(set(images)) != len(images):
        raise HTTPException(status_code=422, detail="the dataset contains duplicates")

    dupe = (await db.execute(
        select(TrainingJob).where(
            TrainingJob.character == body.character,
            TrainingJob.version == body.version,
            TrainingJob.status.not_in(list(TRAINING_TERMINAL)),
        )
    )).scalar_one_or_none()
    if dupe:
        # The database would refuse this anyway via the partial unique index; catching it here
        # turns a 500 into a sentence.
        raise HTTPException(
            status_code=409,
            detail=f"{body.character} v{body.version} is already {dupe.status}. "
                   f"Cancel it, or pick another version.")

    job = TrainingJob(
        user_id=user.id,
        character=body.character,
        trigger=body.trigger,
        version=body.version,
        dataset_images=images,
        config={**RECIPE_DEFAULTS, "steps": body.steps, "caption": body.caption,
                "lora_name": body.lora_name or _default_lora_name(body.character)},
        status=TrainingStatus.PENDING,
        total_steps=body.steps,
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)
    logger.info("queued training %s v%d (%d images) for %s",
                job.character, job.version, len(job.dataset_images), user.username)
    return job


@router.get("/training", response_model=list[TrainingResponse],
            dependencies=[Depends(verify_api_key_or_bearer)])
async def list_training_jobs(
    status: str | None = Query(None),
    limit: int = Query(50, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
):
    q = select(TrainingJob).order_by(TrainingJob.created_at.desc()).limit(limit)
    if status:
        q = q.where(TrainingJob.status == status)
    return list((await db.execute(q)).scalars().all())


@router.get("/training/next", dependencies=[Depends(verify_api_key)])
async def claim_next_training_job(
    worker_id: uuid.UUID = Query(...),
    worker_name: str = Query(None),
    db: AsyncSession = Depends(get_db),
):
    """Hand one queued job to a trainer, or null.

    ONLY A TRAINER MAY CLAIM. Checked first and explicitly: a render worker or a captioner
    picking up a 50-minute training run would be worse than it not being picked up at all.

    Returns null rather than 404 for an empty queue, matching /segments/next -- an empty queue
    is not an error and a poller should not have to distinguish it from one.
    """
    worker = await db.get(Worker, worker_id)
    if worker is None or worker.kind != WorkerKind.TRAINER:
        # Not an error the poller can fix by retrying differently, but not fatal either: a
        # worker that registered before this existed simply gets nothing.
        return None

    await _reclaim_orphans(db)

    # FOR UPDATE SKIP LOCKED, exactly as the segment claim does it: two trainers polling at once
    # must not be handed the same row, and SKIP LOCKED is what makes that true without a queue
    # table or an advisory lock. Oldest first.
    job = (await db.execute(
        select(TrainingJob)
        .where(TrainingJob.status == TrainingStatus.PENDING)
        .order_by(TrainingJob.created_at.asc())
        .limit(1)
        .with_for_update(skip_locked=True)
    )).scalar_one_or_none()
    if job is None:
        return None

    # PRESIGN BEFORE MUTATING. Presigned so the trainer needs no AWS credentials -- the same
    # arrangement the render daemon has for LoRAs -- and the order pairs 1:1 with
    # dataset_images, because the trainer stages them as sel_000..N.
    #
    # Done first because it can fail. Marking the row claimed and then throwing would leave a
    # job owned by a worker that never received the answer: the lost-claim-response failure
    # (wanly-api#242), which the reclaim rules above do eventually fix, but only after the
    # grace period, with the queue stopped in the meantime. Nothing is written unless the
    # response can actually be built.
    try:
        urls = await asyncio.to_thread(
            lambda: [s3.generate_presigned_url(u) for u in job.dataset_images])
    except Exception as e:
        logger.error("could not presign the dataset for %s v%d: %s",
                     job.character, job.version, e)
        raise HTTPException(
            status_code=503,
            detail=f"could not presign the dataset ({e}); the job is still queued") from e

    job.status = TrainingStatus.CLAIMED
    job.worker_id = worker_id
    job.worker_name = worker_name or worker.friendly_name
    # Snapshotted, because the workers row vanishes when a pod drains and "which GPU trained
    # this" is exactly the question asked six weeks later.
    job.gpu_name = (worker.gpu_stats or {}).get("gpu_name")
    job.claimed_at = datetime.now(timezone.utc)
    await db.commit()
    await db.refresh(job)
    logger.info("training %s v%d claimed by %s", job.character, job.version, job.worker_name)
    return TrainingClaimResponse(**TrainingResponse.model_validate(job).model_dump(),
                                 download_urls=urls)


async def _reclaim_orphans(db: AsyncSession) -> None:
    """Put back claims nobody is working on.

    The same three rules the segment claim uses, and for the same reasons -- see
    app/routes/segments.py, which explains at length why reclaiming from a LIVE worker needs all
    three conditions and what happened the time it did not.

    The one that matters most: an empty progress_log is the only trustworthy evidence that a
    claim is not being worked. Status can lie -- a daemon's status push can fail -- but a run
    that is actually going writes progress within seconds.
    """
    now = datetime.now(timezone.utc)
    heartbeat_cutoff = now - timedelta(minutes=STALE_HEARTBEAT_MINUTES)
    orphan_cutoff = now - timedelta(minutes=ORPHANED_TRAINING_MINUTES)

    rows = (await db.execute(
        select(TrainingJob, Worker.last_heartbeat)
        .outerjoin(Worker, Worker.id == TrainingJob.worker_id)
        .where(
            TrainingJob.status.in_(_live_states()),
            TrainingJob.claimed_at.is_not(None),
            or_(
                # dead worker
                Worker.last_heartbeat < heartbeat_cutoff,
                # the worker row is gone entirely
                and_(TrainingJob.claimed_at < orphan_cutoff, Worker.id.is_(None)),
                # alive, idle, and has written nothing: the claim response was lost
                and_(
                    Worker.status.in_(["online", "online-idle"]),
                    Worker.last_heartbeat >= heartbeat_cutoff,
                    or_(TrainingJob.progress_log.is_(None), TrainingJob.progress_log == ""),
                    TrainingJob.claimed_at < orphan_cutoff,
                ),
            ),
        )
    )).all()

    for job, _ in rows:
        logger.warning("reclaiming orphaned training %s v%d from %s",
                       job.character, job.version, job.worker_name)
        job.status = TrainingStatus.PENDING
        job.worker_id = None
        job.worker_name = None
        job.claimed_at = None
        job.progress_log = None
    if rows:
        await db.commit()


@router.get("/training/{job_id}", response_model=TrainingResponse,
            dependencies=[Depends(verify_api_key_or_bearer)])
async def get_training_job(job_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    job = await db.get(TrainingJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    return job


@router.patch("/training/{job_id}", response_model=TrainingResponse,
              dependencies=[Depends(verify_api_key)])
async def update_training_job(
    job_id: uuid.UUID,
    body: TrainingProgress,
    db: AsyncSession = Depends(get_db),
):
    """A trainer reporting in.

    Every field is written only when present. A report that omits a field must never blank what
    an earlier one set -- the mistake the worker heartbeat had to fix twice, where an older
    daemon's partial report erased a newer one's good data.
    """
    job = await db.get(TrainingJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")

    for field in ("progress_log", "step", "total_steps", "error_message",
                  "checkpoints", "output_lora_path"):
        value = getattr(body, field)
        if value is not None:
            setattr(job, field, value)

    # A CANCEL IS STICKY.
    #
    # Nothing here reaches into the GPU box, so a cancelled job keeps reporting `running` until
    # the trainer notices -- and an unconditional write meant every one of those reports undid
    # the cancel. The button appeared to work, the row flipped back within seconds, and the run
    # went to completion. Cancelling had no effect at all.
    #
    # Sticky rather than "only a worker may leave cancelled" because a cancel is a human
    # decision about work nobody wants any more. A trainer that finishes before it notices has
    # not made the run wanted again.
    #
    # This is also how the trainer FINDS OUT: the response carries the row back, so the reply to
    # the progress report it just sent says cancelled, and it stops. There is no second call.
    if body.status is not None and job.status != TrainingStatus.CANCELLED:
        job.status = body.status
        if body.status in TRAINING_TERMINAL:
            job.completed_at = datetime.now(timezone.utc)
            if body.status == TrainingStatus.COMPLETED:
                await _publish_character(db, job)

    await db.commit()
    await db.refresh(job)
    return job


async def _publish_character(db: AsyncSession, job: TrainingJob) -> None:
    """Make the finished LoRA usable in a recipe.

    Nothing else does this. `GET /loras` lists the bucket, so the FILE is discoverable the
    moment it is written -- but a pose fills its <TRIGGER> placeholder from an LtxCharacter row,
    so without one the LoRA exists and no recipe can reach it.

    Upsert on name: retraining a character replaces which LoRA it points at, which is the whole
    point of a v2. The strengths are left alone if the row exists, because they may have been
    tuned by hand.
    """
    if not job.output_lora_path:
        return
    basename = job.output_lora_path.rsplit("/", 1)[-1]
    existing = (await db.execute(
        select(LtxCharacter).where(LtxCharacter.name == job.character)
    )).scalar_one_or_none()
    if existing:
        existing.char_lora = basename
        existing.trigger = job.trigger
        logger.info("character %s now points at %s", job.character, basename)
    else:
        db.add(LtxCharacter(name=job.character, char_lora=basename, trigger=job.trigger))
        logger.info("created character %s -> %s", job.character, basename)


@router.post("/training/{job_id}/artifact", response_model=TrainingResponse,
             dependencies=[Depends(verify_api_key)])
async def upload_training_artifact(
    job_id: uuid.UUID,
    lora: UploadFile = File(...),
    epoch: int | None = None,
    db: AsyncSession = Depends(get_db),
):
    """Publish a finished LoRA.

    THIS ENDPOINT EXISTS BECAUSE POST /upload NEEDS A JWT. A trainer holds the shared worker API
    key and nothing else, so it cannot use the console's upload path. Without this it would need
    AWS credentials in the container -- which no worker in this system has, deliberately.

    Writing to `character/` IS the registration step. GET /loras lists the bucket live and
    derives kind from that prefix, so there is nothing else to call: the LoRA is offerable to
    every worker the moment the object lands. The LtxCharacter row that makes it reachable from
    a recipe is created when the job reports `completed`.

    IT NEEDS s3:PutObject ON ltx-loras/character/*, WHICH THE ROLE DID NOT HAVE.
    Every other bucket this API touches it also writes to, so nothing else noticed; the only
    ltx-loras policy on `wanly-gpu-registry-ec2` was `s3-ltx-loras-readonly`, named exactly
    what it was. The endpoint therefore 500'd in production from the day it shipped:

        botocore.errorfactory.AccessDenied: ... not authorized to perform: s3:PutObject
        on resource: "arn:aws:s3:::ltx-loras/character/pay_v2_e01.safetensors"

    Granted 2026-09-08 as a separate inline policy, `s3-ltx-loras-write-character`, scoped to
    the one prefix this writes -- separate rather than widening the read policy, so its name
    stays true. A 500 here is the first thing to re-check if that role is ever rebuilt.
    """
    job = await db.get(TrainingJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")

    data = await lora.read()
    # A character LoRA at rank 32 is ~650 MB. Anything tiny is a truncated upload or an error
    # page, and letting it land would put a file in the library that fails at load, inside
    # somebody's render, days later.
    if len(data) < 10 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail=f"refusing a {len(data)} byte LoRA — a rank-32 character LoRA is ~650 MB, "
                   f"so this is a truncated upload")

    # The stem the job was created with. Not derived here: stripping `p@y` gives `py`, while
    # the file this project actually renders with is `pay_...` -- a human read `@` as `a`, and
    # no rule produces that. The TRIGGER keeps the original either way; that is what trained.
    stem = (job.config or {}).get("lora_name") or _default_lora_name(job.character)
    tag = f"_e{epoch:02d}" if epoch is not None else ""
    key = f"character/{stem}_v{job.version}{tag}.safetensors"
    uri = await asyncio.to_thread(s3.upload_bytes, data, key, settings.s3_loras_bucket)

    # EVERY EPOCH IS RECORDED, not just the last. Choosing between them is a judgement made by
    # eye at a fixed seed -- loss does not rank them -- so the console has to be able to offer
    # all of them for download. `output_lora_path` tracks the most recent, which is what the
    # character row points at until someone picks differently.
    # ONLY s3:// SURVIVES. The console turns every entry into a download button pointed at
    # GET /files, which can serve an S3 URI and nothing else. Early trainer builds recorded the
    # container-local output path instead of uploading, so a completed job carries entries like
    # `/loras/p@y/ltx23b-v2/output/p@y_v2-000003.comfy.safetensors` -- carrying those forward
    # puts five buttons on the page and four of them 404. Dropping them here is also the
    # backfill: re-uploading a job's epochs replaces the dead list with the live one.
    existing = [c for c in (job.checkpoints or [])
                if isinstance(c, str) and c.startswith("s3://")]
    if uri not in existing:
        job.checkpoints = existing + [uri]
    job.output_lora_path = uri
    await db.commit()
    await db.refresh(job)
    logger.info("published %s (%.0f MB) for %s v%d",
                uri, len(data) / 1024 ** 2, job.character, job.version)
    return job


@router.post("/training/{job_id}/cancel", response_model=TrainingResponse)
async def cancel_training_job(
    job_id: uuid.UUID,
    _user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    job = await db.get(TrainingJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    if job.status in TRAINING_TERMINAL:
        raise HTTPException(status_code=400, detail=f"already {job.status}")
    # The trainer notices on its next report and stops. Nothing here reaches into the GPU box.
    job.status = TrainingStatus.CANCELLED
    job.completed_at = datetime.now(timezone.utc)
    await db.commit()
    await db.refresh(job)
    return job
