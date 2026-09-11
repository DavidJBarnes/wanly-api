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
import re
import uuid
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app import s3
from app.auth import get_current_user, verify_api_key, verify_api_key_or_bearer
from app.config import settings
from app.database import get_db
from app.enums import TRAINING_TERMINAL, TrainingStatus, WorkerKind, worker_can
from app.models import Dataset, LtxCharacter, TrainingJob, User, Worker
from app.schemas.training import (
    MIN_DATASET_IMAGES, TrainingClaimResponse, TrainingCreate, TrainingNotes,
    TrainingProgress, TrainingResponse,
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
    thumbnail = None
    if body.dataset_id:
        ds = await db.get(Dataset, body.dataset_id)
        if not ds:
            raise HTTPException(status_code=404, detail="Dataset not found")
        images = list(ds.images)
        # The anchor is the face the set was checked against; failing that, the first image.
        thumbnail = ds.anchor_uri or (ds.images[0] if ds.images else None)
    # Checked here rather than on the schema, because it applies to whichever of the two was
    # given -- a schema minimum on dataset_images would reject every dataset_id request.
    if len(images) < MIN_DATASET_IMAGES:
        raise HTTPException(
            status_code=422,
            detail=f"{len(images)} images — at least {MIN_DATASET_IMAGES} are needed")
    if len(set(images)) != len(images):
        raise HTTPException(status_code=422, detail="the dataset contains duplicates")

    # THE JOINT GROUP (#102). Resolved the same way group 0 is: a dataset or a raw list for
    # the images, the trigger rule for the caption, its own num_repeats. ABSENT stays
    # absent -- a half-given second identity (a trigger with no images) would build a
    # half-configured dataset silently, so all-or-nothing is enforced here rather than on
    # the schema, because it straddles the dataset resolution that only the route can do.
    second = None
    if body.second_character:
        second_images = list(body.second_dataset_images)
        second_thumbnail = None
        if body.second_dataset_id:
            sds = await db.get(Dataset, body.second_dataset_id)
            if not sds:
                raise HTTPException(status_code=404, detail="Second dataset not found")
            second_images = list(sds.images)
            second_thumbnail = sds.anchor_uri or (sds.images[0] if sds.images else None)
        if len(second_images) < MIN_DATASET_IMAGES:
            raise HTTPException(
                status_code=422,
                detail=f"{len(second_images)} second-identity images — at least "
                       f"{MIN_DATASET_IMAGES} are needed")
        if len(set(second_images)) != len(second_images):
            raise HTTPException(status_code=422, detail="the second dataset contains duplicates")
        if body.second_dataset_id and body.second_dataset_id == body.dataset_id:
            # One dataset training both identities captions every image per group -- the
            # same file cannot carry two triggers at once, and the parity of images across
            # groups would be accidental rather than intended.
            raise HTTPException(
                status_code=422,
                detail="group 0 and the second identity cannot share one dataset; give each "
                       "its own (the datasets balance through num_repeats, not by sharing)")
        if not body.second_trigger:
            raise HTTPException(
                status_code=422,
                detail=f"the second identity ({body.second_character}) has no trigger; its "
                       f"caption would bind to nothing")
        if body.second_trigger == body.trigger:
            raise HTTPException(
                status_code=422,
                detail="the two identities' triggers must differ — one caption pair cannot "
                       "anchor both faces")
        # THE SAME CAPTION RULE, PER GROUP. Group 0's caption learns "<trigger>,
        # <gender>"; group 1's must say the same thing about its own trigger, or the joint
        # LoRA's second face binds to whatever the text happens to be.
        second_caption = None
        if body.second_gender:
            second_caption = f"{body.second_trigger}, {body.second_gender}"
        elif not body.second_caption:
            # A joint group without a resolved caption would fall back to the TRAINER's
            # per-job caption default -- which carries GROUP 0's trigger. Its face would
            # bind to the wrong person's token and the interference this run exists to
            # escape would arrive through the captions instead. A free caption naming the
            # second trigger is accepted; anything else is refused.
            raise HTTPException(
                status_code=422,
                detail=f"the second identity ({body.second_character}) needs a gender or a "
                       f"caption that names its trigger — otherwise the joint LoRA's second "
                       f"face binds to nothing")
        second = {
            "character": body.second_character,
            "trigger": body.second_trigger,
            "gender": body.second_gender,
            "images": second_images,
            "num_repeats": body.second_num_repeats,
        }
        if second_thumbnail and not thumbnail:
            thumbnail = second_thumbnail

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

    # THE CAPTION MUST CARRY THE TRIGGER, or the trigger is never learned. The first run
    # trained with the caption "man" -- typed literally into a field whose placeholder
    # said "trigger, woman" -- so its trigger word meant nothing to the model and the
    # identity bound to "man" instead. A caption that does not mention the trigger gets it
    # prepended here, the way the trainer's own default ("<trigger>, woman") is shaped.
    caption = (body.caption or "").strip() or None
    if body.gender:
        caption = f"{body.trigger}, {body.gender}"
    elif caption and body.trigger not in caption:
        caption = f"{body.trigger}, {caption}"

    # A joint run's steps field is the TOTAL across both groups: the trainer's epoch math
    # reads images x repeats across the two datasets. The combined set is bigger, so the
    # same steps value means fewer passes over each image -- the console states per-image
    # epochs in its estimate, so that is visible to the caller rather than papered over.
    # The dialog scales its default up for a joint run; the API holds one number for both.
    steps = body.steps

    job = TrainingJob(
        user_id=user.id,
        character=body.character,
        trigger=body.trigger,
        version=body.version,
        dataset_images=images,
        config={**RECIPE_DEFAULTS, "steps": steps, "caption": caption,
                "second_caption": second_caption if second else None,
                "gender": body.gender,
                "lora_name": body.lora_name or _default_lora_name(body.character),
                "publish": body.publish},
        second_identity=second,
        status=TrainingStatus.PENDING,
        total_steps=steps,
        thumbnail_uri=thumbnail,
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
    # `worker_can`, not `kind ==`: a box that runs the render stack and the trainer in one
    # container registers as ["render", "trainer"] with kind = render (wanly-gpu-docker#83).
    if worker is None or not worker_can(worker, WorkerKind.TRAINER):
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
        # THE JOINT GROUP, PRESIGNED WITH GROUP 0. One presign pass so a failure in either
        # leaves the row untouched -- the lost-claim-response rule above applies to the
        # second dataset the same as the first, and a joint run that delivered group 0
        # only would stage a single-identity dataset silently, which is worse than a 503.
        second_urls = None
        if job.second_identity:
            second_urls = await asyncio.to_thread(
                lambda: [s3.generate_presigned_url(u) for u in job.second_identity["images"]])
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
    claim = TrainingClaimResponse(**TrainingResponse.model_validate(job).model_dump(),
                                  download_urls=urls)
    if job.second_identity:
        claim.second_download_urls = second_urls
        claim.second_caption = (job.config or {}).get("second_caption")
        claim.second_num_repeats = job.second_identity.get("num_repeats")
    return claim


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
                  "checkpoints", "output_lora_path", "loss_log", "epochs"):
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


@router.patch("/training/{job_id}/notes", response_model=TrainingResponse)
async def set_training_notes(
    job_id: uuid.UUID,
    body: TrainingNotes,
    _user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Operator notes on a run, whole-field replace (wanly-console#484).

    ITS OWN ROUTE, NOT A FIELD ON THE TRAINER'S PATCH, and that is the whole design: the
    trainer's PATCH writes only fields it was handed (a report that omits a field must never
    blank what an earlier one set) and is keyed to the shared worker API key. A note is a
    human write that the reports must never be able to undo, so it keys to a console JWT
    instead and accepts an explicit blank as a deliberate clear.
    """
    job = await db.get(TrainingJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    job.notes = body.notes
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

    The gender travels with the LoRA, not the row: it is the caption THIS file trained on, and
    a v2 captioned differently must render differently (wanly-console#487). A run that recorded
    none leaves the row's alone -- it may have been set by hand for a LoRA that predates it.
    """
    if not job.output_lora_path:
        return
    # THE STEM, NOT THE FILENAME. Every character row stores `pay_v2_e05`, the console's Use
    # button writes the stem, and the character editor says "without .safetensors" -- the
    # first auto-published character stored `david_v1_final.safetensors` and the page could
    # not tell it was in use.
    basename = job.output_lora_path.rsplit("/", 1)[-1].removesuffix(".safetensors")
    existing = (await db.execute(
        select(LtxCharacter).where(LtxCharacter.name == job.character)
    )).scalar_one_or_none()
    gender = (job.config or {}).get("gender") or None
    # A JOINT RUN (#102) publishes its SECOND identity's trigger too. The row carries ONE
    # trigger and ONE gender slot, and a joint LoRA trained on two caption pairs must
    # announce BOTH PAIRS — the phrase the render path fills <TRIGGER> with is
    # "<trigger>, <gender>" (wanly-console#487), so "p@y & d@vid" + the group-0 gender
    # would render "d@vid, woman", a token pair that was never trained: group 1's face
    # bound to "man". The row's trigger becomes the full joint phrase, "p@y, woman &
    # d@vid, man" — exactly what the captions taught, in the order the datasets trained —
    # and the gender slot is left holding the phrase too (readers that use the bare
    # trigger get the same tokens). The joint phrase rides the trigger column because
    # trigger_phrase() concatenates trigger + gender onto whatever the row carries; a
    # separate "both pairs" column would be a second read of the same shape.
    joint = job.second_identity
    if joint:
        g0 = (job.config or {}).get("gender")
        g1 = joint.get("gender")
        trigger = f"{job.trigger}, {g0} & {joint['trigger']}, {g1}"
        gender = None
    else:
        trigger = job.trigger
    if existing:
        existing.char_lora = basename
        existing.trigger = trigger
        if gender:
            existing.gender = gender
        if job.thumbnail_uri:
            existing.image_uri = job.thumbnail_uri
        logger.info("character %s now points at %s", job.character, basename)
    else:
        db.add(LtxCharacter(name=job.character, char_lora=basename, trigger=trigger,
                            gender=gender, image_uri=job.thumbnail_uri))
        logger.info("created character %s -> %s", job.character, basename)


#: Below this a "LoRA" is a truncated upload or an error page. A rank-32 character LoRA is
#: ~650 MB; letting a tiny one land puts a file in the library that fails at load, inside
#: somebody's render, days later.
MIN_ARTIFACT_BYTES = 10 * 1024 * 1024


def _artifact_key(job: TrainingJob, epoch: int | None, final: bool) -> str:
    """Where a checkpoint of this job lives in the LoRA bucket.

    The stem is the one the job was created with, not derived here: stripping `p@y` gives
    `py`, while the file this project actually renders with is `pay_...` -- a human read `@`
    as `a`, and no rule produces that.

    `_eNN` for an epoch, `_final` for the checkpoint written when the step count ran out. The
    trainer writes that one WITHOUT an epoch number, and it is usually a partial epoch -- 1200
    steps over 27 images is 4.4 epochs, so the final file is the fifth, unfinished pass and
    also the one with the most training in it. Left unlabelled it was published as a bare
    `pay_v2.safetensors`, which reads as "the v2" and hides that four other candidates exist.
    """
    stem = (job.config or {}).get("lora_name") or _default_lora_name(job.character)
    tag = "_final" if final else (f"_e{epoch:02d}" if epoch is not None else "")
    return f"character/{stem}_v{job.version}{tag}.safetensors"


def _belongs_to(job: TrainingJob, uri: str) -> bool:
    """Is this URI one of the names `_artifact_key` can produce for THIS job?

    Anchored on the whole name, not a prefix: `pay_v2` is a prefix of `pay_v20_e01`.
    """
    own = f"s3://{settings.s3_loras_bucket}/" + _artifact_key(job, None, False)[:-len(".safetensors")]
    if not uri.startswith(own):
        return False
    return re.fullmatch(r"(_e\d{2}|_final)?\.safetensors", uri[len(own):]) is not None


def _record_checkpoint(job: TrainingJob, uri: str) -> None:
    """One more downloadable checkpoint on the job.

    EVERY EPOCH IS RECORDED, not just the last. Choosing between them is a judgement made by
    eye at a fixed seed -- loss does not rank them -- so the console has to be able to offer
    all of them. `output_lora_path` tracks the most recent, which is what the character row
    points at until someone picks differently.

    ONLY s3:// SURVIVES. The console turns every entry into a download button pointed at
    GET /files, which can serve an S3 URI and nothing else. Early trainer builds recorded the
    container-local output path instead of uploading, so a completed job carried entries like
    `/loras/p@y/ltx23b-v2/output/p@y_v2-000003.comfy.safetensors` -- five buttons on the page
    and four of them 404. Dropping them here is also the backfill.
    """
    existing = [c for c in (job.checkpoints or [])
                if isinstance(c, str) and c.startswith("s3://")]
    if uri not in existing:
        # Reassigned, never appended: JSONB does not see an in-place mutation.
        job.checkpoints = existing + [uri]
    job.output_lora_path = uri


@router.post("/training/{job_id}/artifact-url", dependencies=[Depends(verify_api_key)])
async def presign_training_artifact(
    job_id: uuid.UUID,
    epoch: int | None = None,
    final: bool = False,
    db: AsyncSession = Depends(get_db),
):
    """Where to PUT one checkpoint, signed so the trainer needs no AWS credentials.

    THE TRAINER WRITES STRAIGHT TO S3. The multipart `/artifact` endpoint below still works
    and is what a CLI backfill uses, but it was the wrong shape for the trainer: five 650 MB
    files, each read whole into the memory of a t3.small and re-sent to S3, one after the
    other, AFTER training -- so the console showed "running" at 100% for the hour that took,
    and when two of the five did not survive the trip the job still said "completed" with
    the final checkpoint missing and the character row pointing at last week's file.

    Two calls: this one for the URL, `/artifact-commit` once the PUT succeeded. The commit is
    what records the checkpoint, and it checks the object is really there and really a LoRA,
    so a failed or truncated PUT records nothing.
    """
    job = await db.get(TrainingJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    if epoch is None and not final:
        raise HTTPException(status_code=422, detail="say which: epoch=N or final=true")
    uri = f"s3://{settings.s3_loras_bucket}/{_artifact_key(job, epoch, final)}"
    try:
        url = await asyncio.to_thread(s3.generate_presigned_put, uri)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"could not presign {uri}: {e}") from e
    return {"uri": uri, "put_url": url, "expires_in": 21600}


@router.post("/training/{job_id}/artifact-commit", response_model=TrainingResponse,
             dependencies=[Depends(verify_api_key)])
async def commit_training_artifact(
    job_id: uuid.UUID,
    uri: str = Query(...),
    db: AsyncSession = Depends(get_db),
):
    """The trainer says a checkpoint landed. Believed only after looking.

    The object has to exist, be big enough to be a LoRA, and belong to THIS job -- a trainer
    cannot register somebody else's file, or a file outside `character/`, against a run.
    """
    job = await db.get(TrainingJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    if not _belongs_to(job, uri):
        raise HTTPException(
            status_code=422,
            detail=f"{uri} is not a checkpoint of {job.character} v{job.version}")
    head = await asyncio.to_thread(s3.head_object, uri)
    if head is None:
        raise HTTPException(status_code=409, detail=f"{uri} is not in the bucket — the PUT did not land")
    if head["Size"] < MIN_ARTIFACT_BYTES:
        raise HTTPException(
            status_code=409,
            detail=f"{uri} is {head['Size']} bytes — a rank-32 character LoRA is ~650 MB, so "
                   f"this is a truncated upload")
    # AND REALLY A LORA, not just big enough to be one. See s3.safetensors_header_ok for the
    # zero-filled final that a size check waved through.
    if not await asyncio.to_thread(s3.safetensors_header_ok, uri):
        raise HTTPException(
            status_code=422,
            detail=f"{uri} is in the bucket but has no safetensors header (a zero-filled or "
                   f"truncated file). Not recording it; the trainer should regenerate it.")
    _record_checkpoint(job, uri)
    await db.commit()
    await db.refresh(job)
    logger.info("recorded %s (%.0f MB) for %s v%d",
                uri, head["Size"] / 1024 ** 2, job.character, job.version)
    return job


@router.post("/training/{job_id}/artifact", response_model=TrainingResponse,
             dependencies=[Depends(verify_api_key)])
async def upload_training_artifact(
    job_id: uuid.UUID,
    lora: UploadFile = File(...),
    epoch: int | None = None,
    final: bool = False,
    db: AsyncSession = Depends(get_db),
):
    """Publish a finished LoRA, through this API. The CLI and backfill path.

    The trainer no longer uses this -- see `/artifact-url` for why -- but a curl from a
    laptop with a checkpoint on it still does, and it has to land in the same place with the
    same name.

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
    if len(data) < MIN_ARTIFACT_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"refusing a {len(data)} byte LoRA — a rank-32 character LoRA is ~650 MB, "
                   f"so this is a truncated upload")
    key = _artifact_key(job, epoch, final)
    uri = await asyncio.to_thread(s3.upload_bytes, data, key, settings.s3_loras_bucket)
    _record_checkpoint(job, uri)
    await db.commit()
    await db.refresh(job)
    logger.info("published %s (%.0f MB) for %s v%d",
                uri, len(data) / 1024 ** 2, job.character, job.version)
    return job


@router.post("/training/{job_id}/publish", response_model=TrainingResponse)
async def request_publish(
    job_id: uuid.UUID,
    label: str = Query(..., pattern=r"^(e\d{2}|final)$"),
    _user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Ask for a checkpoint that stayed on the trainer to be uploaded after all.

    Only the final checkpoint goes up by default ("I often only want 1 or 2 epochs"); the
    rest sit in the run directory on the GPU box. The trainer polls its finished jobs for
    this list and uploads what it finds, the same way as during a run.
    """
    job = await db.get(TrainingJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    if not any(e.get("label") == label for e in (job.epochs or [])):
        raise HTTPException(status_code=404, detail=f"this run has no checkpoint {label}")
    tag = f"_{label}.safetensors"
    if any(c.endswith(tag) for c in (job.checkpoints or [])):
        raise HTTPException(status_code=409, detail=f"{label} is already in the bucket")
    wanted = list(job.publish_requests or [])
    if label not in wanted:
        job.publish_requests = wanted + [label]
        await db.commit()
        await db.refresh(job)
    return job


@router.delete("/training/{job_id}", status_code=204)
async def delete_training_job(
    job_id: uuid.UUID,
    purge: bool = True,
    _user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Take a finished run off the board, and its LoRA files out of the bucket.

    Terminal jobs only: a live one is cancelled first, so that the trainer working on it
    still has a row to report to.

    THE FILES GO TOO, by default. A deleted run whose checkpoints linger in the library is
    what someone deleting a run does not expect (wanly-console#464), and a LoRA nobody can
    trace to a run is a LoRA nobody dares delete later. The one thing that stops it: a
    character that currently renders with one of these checkpoints. Deleting under it would
    break every recipe that names the character, so that is refused and says which.
    """
    job = await db.get(TrainingJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    if job.status not in TRAINING_TERMINAL:
        raise HTTPException(status_code=409, detail=f"still {job.status} — cancel it first")
    files = [c for c in (job.checkpoints or []) if isinstance(c, str) and c.startswith("s3://")]
    if purge and files:
        stems = {f.rsplit("/", 1)[-1].removesuffix(".safetensors") for f in files}
        using = (await db.execute(
            select(LtxCharacter).where(LtxCharacter.char_lora.in_(stems))
        )).scalars().all()
        if using:
            names = ", ".join(f"{c.name} ({c.char_lora})" for c in using)
            raise HTTPException(
                status_code=409,
                detail=f"{names} renders with a checkpoint of this run — point the character "
                       f"at another LoRA first, or delete the run without its files")
        try:
            await asyncio.gather(*(asyncio.to_thread(s3.delete_object, f) for f in files))
        except Exception as e:
            # The role needs s3:DeleteObject on ltx-loras/character/*, which it did not have
            # the first time this ran -- the read policy is read-only by name and the write
            # policy granted PutObject alone. A 500 here left the row in place and said
            # nothing; say what is needed instead.
            raise HTTPException(
                status_code=503,
                detail=f"could not delete the LoRA files ({type(e).__name__}: {e}); the run "
                       f"is still here. The API's role needs s3:DeleteObject on the "
                       f"character/ prefix.") from e
        logger.info("deleted %d checkpoint(s) of %s v%d", len(files), job.character, job.version)
    await db.delete(job)
    await db.commit()


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
