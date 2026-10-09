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

from fastapi import APIRouter, Depends, File, HTTPException, Path, Query, UploadFile
from sqlalchemy import String, and_, cast, func, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app import run_datasets, s3
from app.auth import get_current_user, verify_api_key, verify_api_key_or_bearer
from app.config import settings
from app.database import get_db
from app.enums import TRAINING_TERMINAL, TrainingStatus, WorkerKind, worker_can
from app.models import (LtxCharacter, Segment, TrainingJob, User, Worker,
                        training_arch)
from app.character_registry import identity_phrase
from app.schemas.training import (
    TrainedOn, TrainedOnGroup, TrainingClaimResponse, TrainingCreate,
    TrainingNotes, TrainingPreflight, TrainingProgress, TrainingResponse,
)
from app.training_plan import REG_RATIO, plan_training

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
#: SDXL (#398): the "aio" recipe, the values the trainer is told to use. Mirrors the
#: trainer's SDXL_DEFAULTS; snapshotted for the same reason as the LTX set.
SDXL_RECIPE_DEFAULTS = {
    "network_dim": 128,
    "network_alpha": 64,
    "learning_rate": 8e-5,
    "text_encoder_lr": 2e-5,
    "num_repeats": 8,
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


@router.post("/training/preflight", response_model=TrainingPreflight)
async def preflight_training_job(
    body: TrainingCreate,
    _user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """What POST /training would do with this body, and everything that stops it (#352).

    The checklist the Train dialog renders: every problem at once (not the first), the
    groups the server derived -- datasets, repeats, and the first final captions of each,
    trigger prefix included -- and the epoch arithmetic. Writes nothing. The same function
    decides POST /training, so a green checklist is a run the API will accept.
    """
    return (await plan_training(db, body)).public()


@router.post("/training", response_model=TrainingResponse, status_code=201)
async def create_training_job(
    body: TrainingCreate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Queue a training run. The console's entry point.

    EVERYTHING BUT WHO IS DERIVED (#352) -- see app/training_plan.py. A request with any
    problem is refused whole, 422 {detail: {problems, warnings}}, with the same list the
    preflight shows.

    CAPTIONS ARE SNAPSHOTTED HERE. Each group records its final per-image captions beside
    its images, so the run trains on the words that were previewed -- a caption edited
    after the Train button, or an owner's trigger re-registered, cannot change what a
    queued job does. /retry reads the snapshot too, never the live captions.

    GROUP 0 is the first member (solo: the character) and lives in the flat columns and
    config, exactly where the trainer has always read it; groups 1..N -- the second member,
    the composition set, the regularization pools -- go in `identities`.
    """
    plan = await plan_training(db, body)
    if not plan.ok:
        raise HTTPException(status_code=422,
                            detail={"problems": plan.problems, "warnings": plan.warnings})
    g0, rest = plan.groups[0], plan.groups[1:]
    # The anchor is the face the set was checked against; failing that, the first image.
    ds0 = g0.dataset
    thumbnail = (ds0.anchor_uri if ds0 and ds0.anchor_uri else None) or \
        (g0.images[0] if g0.images else None)
    job = TrainingJob(
        user_id=user.id,
        character=body.character,
        # Group 0's trigger. For a pair the row's trigger is the joined phrase, but that is
        # the PUBLISHED phrase, assembled at publish from every identity group; this column
        # has always been group 0's token, and the trainer reads it that way.
        trigger=g0.trigger,
        version=body.version,
        dataset_images=g0.images,
        config={**(SDXL_RECIPE_DEFAULTS if body.arch == "sdxl" else RECIPE_DEFAULTS),
                "arch": body.arch,
                "seed": RECIPE_DEFAULTS["seed"] if body.seed is None else body.seed,
                "steps": body.steps,
                "num_repeats": g0.num_repeats,
                "mode": body.mode,
                "members": [m.name for m in plan.members],
                "kind": g0.kind,
                "gender": g0.gender,
                # The legacy single caption, for a trainer that predates `captions`: it
                # would train every image under the first one, which is wrong but not
                # silent -- the per-image list is the contract.
                "caption": g0.captions[0] if g0.captions else None,
                "captions": g0.captions,
                "base_checkpoint": plan.base_checkpoint,
                "reg_ratio": REG_RATIO if body.regularization else 0,
                "caption_mode": body.caption_mode,
                "allow_no_composition": body.allow_no_composition,
                "allow_low_scores": body.allow_low_scores,
                "dataset": g0.provenance(),
                "lora_name": body.lora_name or _default_lora_name(body.character),
                "publish": body.publish},
        identities=[_group_row(g) for g in rest] or None,
        status=TrainingStatus.PENDING,
        total_steps=body.steps,
        thumbnail_uri=thumbnail,
    )
    db.add(job)
    try:
        await db.flush()
        # THE RUN IS THE RECORD (#422). One row per group, from the snapshot just written, in
        # the same transaction: a run never exists without its record of what it trained on.
        db.add_all(run_datasets.rows_for_job(job))
        await db.commit()
    except IntegrityError:
        # The live-version unique index racing another create between the plan's check
        # and this commit.
        await db.rollback()
        raise HTTPException(
            status_code=409,
            detail=f"{body.character} v{body.version} already has a live run") from None
    await db.refresh(job)
    logger.info("queued %s training %s v%d: %s for %s", body.mode, job.character, job.version,
                ", ".join(f"{g.kind} {g.dataset.name if g.dataset else '?'} "
                          f"{len(g.images)}x{g.num_repeats}" for g in plan.groups),
                user.username)
    return job


def _group_row(g) -> dict:
    """One extra group as the job stores it: the snapshot a claim and a retry both read."""
    return {
        "character": g.character,
        "trigger": g.trigger,
        "gender": g.gender,
        "kind": g.kind,
        "caption": g.captions[0] if g.captions else None,
        "captions": g.captions,
        "images": g.images,
        "num_repeats": g.num_repeats,
        #: Samples per item per repeat: CLIP_WINDOWS for a clip group (#411), else 1.
        "windows": g.windows,
        #: Provenance, snapshot at creation: a rename must not rewrite what trained.
        "dataset": g.provenance(),
    }


@router.get("/training", response_model=list[TrainingResponse],
            dependencies=[Depends(verify_api_key_or_bearer)])
async def list_training_jobs(
    status: str | None = Query(None),
    #: One character's runs, every arch (#404): the character card's versions table. Exact
    #: name match -- TrainingJob.character is the registry name, not a foreign key.
    character: str | None = Query(None, max_length=64),
    limit: int = Query(50, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
):
    q = select(TrainingJob).order_by(TrainingJob.created_at.desc()).limit(limit)
    if status:
        q = q.where(TrainingJob.status == status)
    if character:
        q = q.where(TrainingJob.character == character)
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
        # THE EXTRA GROUPS, PRESIGNED WITH GROUP 0. One presign pass so a failure anywhere
        # leaves the row untouched -- the lost-claim-response rule above applies to every
        # dataset the same as the first, and a joint run that delivered group 0 only would
        # stage a single-identity dataset silently, which is worse than a 503.
        group_payload = []
        for g in (job.identities or []):
            g_urls = await asyncio.to_thread(
                lambda imgs=g["images"]: [s3.generate_presigned_url(u) for u in imgs])
            group_payload.append({
                "character": g.get("character"),
                "trigger": g.get("trigger"),
                "gender": g.get("gender"),
                "caption": g.get("caption"),
                #: Per-image, parallel to download_urls (#352). None on a pre-#352 job,
                #: whose single `caption` is the whole contract.
                "captions": g.get("captions"),
                #: identity | composition | regularization. Recorded since #352; before it
                #: a group without a trigger was a composition group, which is exactly
                #: how the trainer read it then.
                "kind": g.get("kind") or ("identity" if g.get("trigger") else "composition"),
                "num_repeats": g.get("num_repeats"),
                #: Windows per clip for a clip group (#411); absent means 1.
                "windows": g.get("windows") or 1,
                #: So the trainer can say WHICH dataset it is staging, not just how many.
                "dataset_name": (g.get("dataset") or {}).get("name"),
                "download_urls": g_urls,
            })
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
    # `identities` is present in the base dump because TrainingResponse declares it, and the
    # claim re-declares it as the resolved, presigned override. Passing both collides on the
    # keyword and 500s on EVERY claim (#322), so the base's copy is dropped and the resolved
    # one wins. Exercised by TestTheClaimEndpoint below.
    payload = TrainingResponse.model_validate(job).model_dump()
    payload.pop("identities", None)
    claim = TrainingClaimResponse(**payload, download_urls=urls,
                                  captions=(job.config or {}).get("captions"),
                                  identities=group_payload or None)
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


@router.get("/training/{job_id}/home", dependencies=[Depends(verify_api_key_or_bearer)])
async def get_run_home(job_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """{dataset_id}: the dataset page a run lives on (wanly-console#647) -- its composition set
    for a pair, else its first set; for an orphan (every set gone), its character's living
    set. dataset_id is None when none of those exists. Old /training links resolve through
    this now that training is reached through datasets only."""
    job = await db.get(TrainingJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    links = await run_datasets.link_rows(db, job)
    home = run_datasets.home_dataset_id(links, job)
    if home is None:
        home = run_datasets.orphan_home(job, links, await run_datasets._set_summaries(db))
    return {"dataset_id": str(home) if home else None}


@router.get("/training/{job_id}/trained-on", response_model=TrainedOn,
            dependencies=[Depends(verify_api_key_or_bearer)])
async def get_trained_on(job_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """What this run trained on, per dataset, and how each set differs now (#422).

    THE RUN IS THE RECORD (#419): sets stay editable after training, so the answer to "which
    images were in v2" lives here -- the images and captions exactly as trained, plus what
    has been added to or removed from each set since.
    """
    job = await db.get(TrainingJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    return TrainedOn(job_id=str(job.id), character=job.character, version=job.version,
                     arch=((job.config or {}).get("arch") or "ltx"),
                     groups=[TrainedOnGroup(**g) for g in await run_datasets.trained_on(db, job)])


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
    # A DELETED CHECKPOINT STAYS DELETED (#413). The trainer reports what is on its disk, and
    # until its poll has removed the files that includes labels already deleted here.
    gone = set(job.delete_requests or [])
    if gone and body.epochs is not None:
        job.epochs = [e for e in body.epochs if e.get("label") not in gone] or None

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
                # NOT a publish any more (wanly-api#452): which checkpoint a character
                # renders with is the user's STAR, never "whichever run finished last". A
                # pair's first run still creates its row, as a draft, so its page exists.
                await _register_on_finish(db, job)

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


async def _register_on_finish(db: AsyncSession, job: TrainingJob) -> None:
    """A finished run's only effect on the registry (wanly-api#452): a PAIR trained for the
    first time gets its row -- kind, members, the joined trigger phrase -- with NO LoRA yet,
    a draft until a checkpoint is starred. A solo character's row exists before it trains,
    and an SDXL run never touches the registry."""
    cfg = job.config or {}
    if cfg.get("arch") == "sdxl" or cfg.get("mode") != "pair":
        return
    existing = (await db.execute(
        select(LtxCharacter).where(LtxCharacter.name == job.character))).scalar_one_or_none()
    if existing is not None:
        return
    stamp = _pair_stamp(job)
    db.add(LtxCharacter(name=job.character, char_lora=None,
                        trained_from=_trained_from(job), **stamp))
    logger.info("created pair %s (draft: no starred checkpoint yet)", job.character)


def _pair_stamp(job: TrainingJob) -> dict:
    """kind/members/trigger/gender of a pair row, from the run's own snapshot."""
    cfg = job.config or {}
    phrases = [identity_phrase(job.trigger, cfg.get("gender"))]
    phrases += [identity_phrase(g.get("trigger"), g.get("gender"))
                for g in (job.identities or []) if g.get("kind") == "identity"]
    return {"kind": "pair", "members": list(cfg.get("members") or []),
            "trigger": " and ".join(p for p in phrases if p), "gender": None}


def checkpoint_uri(job: TrainingJob, label: str) -> str | None:
    """The uploaded checkpoint `label` (e01.. or final) of this run, or None if it is not in
    the bucket (it may live only on the trainer: publish "none" is the default, #413)."""
    tag = f"_{label}.safetensors"
    return next((c for c in (job.checkpoints or []) if isinstance(c, str) and c.endswith(tag)),
                None)


async def star_checkpoint(db: AsyncSession, c: LtxCharacter, job: TrainingJob, uri: str) -> None:
    """Make `uri` (one of `job`'s uploaded LTX checkpoints) the character's LoRA: char_lora,
    provenance, base model -- and for a pair its phrase -- exactly as a publish stamped them.
    Clears any pending star."""
    await _publish_character(db, job, uri=uri, row=c)
    c.star_pending = None


async def apply_pending_stars(db: AsyncSession, job: TrainingJob, uri: str) -> None:
    """A checkpoint just landed: any character that starred it while it was still on the
    trainer now points at it (wanly-api#452)."""
    label = uri.rsplit("_", 1)[-1].removesuffix(".safetensors")
    rows = (await db.execute(select(LtxCharacter).where(
        LtxCharacter.star_pending["training_job_id"].astext == str(job.id),
        LtxCharacter.star_pending["label"].astext == label))).scalars().all()
    for c in rows:
        await star_checkpoint(db, c, job, uri)
        logger.info("character %s: pending star %s applied -> %s", c.name, label, uri)


async def _publish_character(db: AsyncSession, job: TrainingJob, uri: str | None = None,
                             row: LtxCharacter | None = None) -> None:
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
    uri = uri or job.output_lora_path
    if not uri:
        return
    if (job.config or {}).get("arch") == "sdxl":
        # A START-IMAGE LoRA (#398). The LTX engine cannot load it, so pointing a character
        # row at it would break every render of that character. It is downloaded from the run.
        return
    if (job.config or {}).get("mode") in ("solo", "pair"):
        await _publish_registered(db, job, uri=uri, row=row)
        return
    # ---- A PRE-#352 JOB, published exactly as it always was. Retrying an old failed run
    # must not change what it publishes to.
    #
    # THE STEM, NOT THE FILENAME. Every character row stores `pay_v2_e05`, the console's Use
    # button writes the stem, and the character editor says "without .safetensors" -- the
    # first auto-published character stored `david_v1_final.safetensors` and the page could
    # not tell it was in use.
    basename = uri.rsplit("/", 1)[-1].removesuffix(".safetensors")
    existing = row or (await db.execute(
        select(LtxCharacter).where(LtxCharacter.name == job.character)
    )).scalar_one_or_none()
    gender = (job.config or {}).get("gender") or None
    # A JOINT RUN (#102, #106) publishes EVERY identity's trigger. The row carries ONE
    # trigger and ONE gender slot, and a LoRA trained on several caption pairs must announce
    # them all — the phrase the render path fills <TRIGGER> with is "<trigger>, <gender>"
    # (wanly-console#487), so a bare "p@y and d@vid" + one gender would render a pair that
    # was never trained. The row's trigger becomes the full phrase, "p@y, woman and d@vid,
    # man" — exactly what the captions taught, in the order the groups were given — and the
    # gender slot is left None (the phrase carries them all).
    #
    # Only IDENTITY groups contribute a pair. A COMPOSITION group (#106) has no trigger: its
    # caption repeats pairs already here, and including it would say the same face twice.
    #
    # " and " is natural text: the whole phrase lands in the one <TRIGGER> placeholder and
    # the scene text names who is who. Not "&": the captions never contained it.
    pairs: list[str] = []

    def _add_pair(trig: str | None, gen: str | None) -> None:
        if not trig:
            return
        pair = f"{trig}, {gen}" if gen else trig
        if pair not in pairs:
            pairs.append(pair)

    _add_pair(job.trigger, gender)
    for g in (job.identities or []):
        _add_pair(g.get("trigger"), g.get("gender"))
    if len(pairs) > 1:
        trigger = " and ".join(pairs)
        gender = None
    else:
        trigger = job.trigger
    # THE DATASETS THIS LORA CAME FROM, in group order (migration 099). Built from what the
    # job recorded at creation, not from live dataset rows: a rename must not rewrite what
    # trained, and a deleted dataset keeps its name (its id is then dangling, and the
    # console renders the entry as plain text). Stamped on every publish, so a retrain
    # replaces v1's provenance with v2's.
    trained_from = []
    for raw in [dict((job.config or {}).get("dataset") or {"id": None, "name": None,
                                                           "count": len(job.dataset_images)})
                ] + [dict(g.get("dataset") or {"id": None, "name": None,
                                               "count": len(g.get("images") or [])})
                     for g in (job.identities or [])]:
        trained_from.append({"dataset_id": raw.get("id"), "name": raw.get("name"),
                             "count": raw.get("count")})
    if existing:
        existing.char_lora = basename
        existing.trigger = trigger
        if gender:
            existing.gender = gender
        existing.trained_from = trained_from
        if job.thumbnail_uri:
            existing.image_uri = job.thumbnail_uri
        logger.info("character %s now points at %s", job.character, basename)
    else:
        db.add(LtxCharacter(name=job.character, char_lora=basename, trigger=trigger,
                            gender=gender, image_uri=job.thumbnail_uri,
                            trained_from=trained_from))
        logger.info("created character %s -> %s", job.character, basename)


def _lora_stem(job: TrainingJob) -> str:
    # THE STEM, NOT THE FILENAME: see _publish_character.
    return job.output_lora_path.rsplit("/", 1)[-1].removesuffix(".safetensors")


def _trained_from(job: TrainingJob) -> list[dict]:
    """[{dataset_id, name, count, kind}] in group order, from what creation recorded."""
    cfg = job.config or {}
    rows = [(cfg.get("dataset") or {}, cfg.get("kind") or "identity",
             len(job.dataset_images))]
    rows += [(g.get("dataset") or {}, g.get("kind"), len(g.get("images") or []))
             for g in (job.identities or [])]
    return [{"dataset_id": d.get("id"), "name": d.get("name"),
             "count": d.get("count", n), "kind": kind} for d, kind, n in rows]


async def _publish_registered(db: AsyncSession, job: TrainingJob, uri: str | None = None,
                              row: LtxCharacter | None = None) -> None:
    """Publish a #352 run: to its OWN row, and never to anybody else's.

    SOLO updates the character it trained: the LoRA, provenance, base model and face. The
    trigger and gender are left alone -- the run captioned with the registry's values, so
    they already are what this LoRA learned, and a publish is not the place to change them.

    PAIR upserts the PAIR's row (kind=pair, members) and never touches a member's. A joint
    run used to publish over its first member's solo row, so training "DavidKelly" replaced
    what "David" rendered with -- and "David" then rendered a LoRA that carried two faces.
    The pair row's trigger is the joined phrase of the identity groups in group order,
    "d@vid, man and k3lly2026, woman" -- exactly the prefix the composition captions
    carried, so <TRIGGER> renders the words the two-person frames taught. Its gender is
    None: the phrase carries both. Rebuilt from the job's own snapshot, because that is
    what trained, whatever the registry says now.

    Strengths are left as they are on an existing row: they may have been tuned by hand.
    """
    cfg = job.config or {}
    basename = (uri.rsplit("/", 1)[-1].removesuffix(".safetensors") if uri
                else _lora_stem(job))
    existing = row or (await db.execute(
        select(LtxCharacter).where(LtxCharacter.name == job.character)
    )).scalar_one_or_none()
    stamp = {"char_lora": basename, "trained_from": _trained_from(job),
             "base_checkpoint": cfg.get("base_checkpoint")}
    if cfg.get("mode") == "pair":
        stamp.update(_pair_stamp(job))
    if existing:
        for k, v in stamp.items():
            setattr(existing, k, v)
        if job.thumbnail_uri:
            existing.image_uri = job.thumbnail_uri
        logger.info("%s %s now points at %s", cfg.get("mode"), job.character, basename)
    else:
        # A solo row is registered before it trains, so this is a pair's first publish --
        # or a solo row deleted mid-run, rebuilt from what the run captioned with.
        stamp.setdefault("trigger", job.trigger)
        stamp.setdefault("gender", cfg.get("gender"))
        db.add(LtxCharacter(name=job.character, image_uri=job.thumbnail_uri, **stamp))
        logger.info("created %s character %s -> %s", cfg.get("mode"), job.character, basename)


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
    if (job.config or {}).get("arch") == "sdxl":
        # SDXL (#398): under character/ because that is all the API's role may write, in its
        # own folder so the listing files it as kind "character/sdxl" -- which render
        # workers do NOT sync eagerly (they take kind == "character") -- and with "_sdxl" IN
        # THE NAME, because workers flatten the prefix away and skip BOTH files of a name
        # that appears under two prefixes. Without it k3lly's SDXL v1 would knock her LTX v1
        # out of every worker.
        return f"character/sdxl/{stem}_sdxl_v{job.version}{tag}.safetensors"
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
    # THE FINAL KEEPS IT (#413). Under publish "all" the final goes up first and the epochs
    # still queued follow it, so "the most recent" was an epoch -- and the character row,
    # published from this at completion, pointed at e03 instead of the finished LoRA.
    if uri.endswith("_final.safetensors") or not any(
            c.endswith("_final.safetensors") for c in existing):
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
    # DELETED WHILE IT WAS UPLOADING (#413). The person threw this checkpoint away after the
    # trainer had started its PUT; recording it now would bring it back, unlisted in `epochs`
    # and so with no row to delete it from again. The object goes; the trainer is told.
    deleted = next((lbl for lbl in (job.delete_requests or [])
                    if uri.endswith(f"_{lbl}.safetensors")), None)
    if deleted:
        await asyncio.to_thread(s3.delete_object, uri)
        raise HTTPException(status_code=410, detail=f"{deleted} was deleted; not recording it")
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
    await apply_pending_stars(db, job, uri)
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
    await apply_pending_stars(db, job, uri)
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


#: Segment statuses that will still load a LoRA: a checkpoint they name must not vanish
#: under them (#413). Terminal segments only hold a record of what they used.
_LIVE_SEGMENT = ("pending", "claimed", "processing", "awaiting_caption")


@router.delete("/training/{job_id}/checkpoints/{label}", response_model=TrainingResponse)
async def delete_checkpoint(
    job_id: uuid.UUID,
    label: str = Path(..., pattern=r"^(e\d{2}|final)$"),
    _user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete ONE checkpoint forever: its S3 copy now, its files on the trainer next poll.

    THE POINT IS TO TRY, THEN KEEP OR THROW AWAY (#413). Most epochs are looked at once and
    discarded; with nothing uploading by default they sit on the trainer, and this is the
    other half of that -- the disk does not fill with runs nobody wanted.

    REFUSED while anything still depends on it, because "forever" cannot be taken back:
      * the run is live -- the trainer is still writing and reporting this list;
      * a character renders with it (any character, pairs included, as the run delete);
      * a segment that has not finished names it -- its LoRA is loaded at claim, and a file
        gone from the bucket would fail that render (or every worker's next sync) instead.
    The S3 delete happens before anything is written, so a failure changes nothing.
    """
    job = await db.get(TrainingJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    if job.status not in TRAINING_TERMINAL:
        raise HTTPException(status_code=409,
                            detail=f"still {job.status} — checkpoints can be deleted once it ends")
    if label in (job.delete_requests or []) or not any(
            e.get("label") == label for e in (job.epochs or [])):
        raise HTTPException(status_code=404, detail=f"this run has no checkpoint {label}")

    tag = f"_{label}.safetensors"
    uploaded = [c for c in (job.checkpoints or []) if isinstance(c, str) and c.endswith(tag)]
    for uri in uploaded:
        stem = uri.rsplit("/", 1)[-1].removesuffix(".safetensors")
        using = (await db.execute(
            select(LtxCharacter).where(LtxCharacter.char_lora == stem))).scalars().all()
        if using:
            raise HTTPException(
                status_code=409,
                detail=f"{', '.join(c.name for c in using)} renders with {stem} — point the "
                       f"character at another LoRA first")
        live = (await db.execute(
            select(func.count()).select_from(Segment).where(
                Segment.status.in_(_LIVE_SEGMENT),
                cast(Segment.ltx_recipe, String).contains(stem)))).scalar_one()
        if live:
            raise HTTPException(
                status_code=409,
                detail=f"{live} queued or running segment(s) render with {stem} — let them "
                       f"finish, or remove them, first")
    try:
        await asyncio.gather(*(asyncio.to_thread(s3.delete_object, u) for u in uploaded))
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"could not delete {label} from the bucket ({type(e).__name__}: {e}); "
                   f"nothing was changed") from e

    # Reassigned throughout: JSONB does not see in-place mutation.
    left = [c for c in (job.checkpoints or []) if c not in uploaded]
    job.checkpoints = left or None
    if job.output_lora_path in uploaded or job.output_lora_path not in left:
        finals = [c for c in left if c.endswith("_final.safetensors")]
        job.output_lora_path = (finals or left or [None])[-1]
    job.publish_requests = [p for p in (job.publish_requests or []) if p != label] or None
    job.epochs = [e for e in (job.epochs or []) if e.get("label") != label] or None
    job.delete_requests = list(job.delete_requests or []) + [label]
    await db.commit()
    await db.refresh(job)
    logger.info("deleted checkpoint %s of %s v%d (%s)", label, job.character, job.version,
                "bucket + trainer" if uploaded else "trainer only")
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


@router.post("/training/{job_id}/retry", response_model=TrainingResponse)
async def retry_training_job(
    job_id: uuid.UUID,
    _user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Queue a failed run again, in place, training EXACTLY its snapshot (#423).

    THE RUN IS THE RECORD (#419). api#342 made retry re-read each group's images from its
    dataset, back when a trained set was frozen and the only thing that changed between
    attempts was a dead URI. Datasets are living now (#420): re-reading would quietly retrain
    a different set than the one this run records and was previewed with. So a retry re-queues
    the snapshot -- images, captions, repeats -- unchanged, and its training_run_datasets rows
    stay true. Files a run trained on are never deleted by the dataset paths (#421); to train
    on what a set holds now, start a new run from it.

    In place, like POST /segments/{id}/retry: same row, same character vN, not a new
    version. The attempt's artifacts (loss curve, recorded checkpoints, publish requests)
    are cleared with the progress — the trainer writes the same version-named paths again,
    so a stale list would only mislabel the new attempt's files. The human's notes stay.

    Only FAILED is retryable. A cancelled run is work somebody decided they did not want;
    re-creating it through the dialog is the way to change one's mind about that.
    """
    job = await db.get(TrainingJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    if job.status != TrainingStatus.FAILED:
        raise HTTPException(
            status_code=400,
            detail=f"only failed runs can be retried (this one is {job.status})")

    twin = (await db.execute(
        select(TrainingJob).where(
            TrainingJob.character == job.character,
            TrainingJob.version == job.version,
            # Per arch (#402), as the live unique index is.
            training_arch() == ((job.config or {}).get("arch") or "ltx"),
            TrainingJob.id != job.id,
            TrainingJob.status.not_in(list(TRAINING_TERMINAL)),
        )
    )).scalar_one_or_none()
    if twin:
        raise HTTPException(
            status_code=409,
            detail=f"{job.character} v{job.version} is already {twin.status}. "
                   f"Cancel it first, or retry that one instead of this one.")

    job.status = TrainingStatus.PENDING
    # Back to a pristine queue row, same columns as _reclaim_orphans plus the attempt's
    # outputs and artifacts.
    job.worker_id = None
    job.worker_name = None
    job.claimed_at = None
    job.completed_at = None
    job.progress_log = None
    job.error_message = None
    job.step = None
    job.loss_log = None
    job.epochs = None
    job.checkpoints = None
    job.output_lora_path = None
    job.publish_requests = None
    # The retry writes the SAME version-named files again (#413): an old delete request left
    # here would have the trainer delete the new attempt's checkpoint of that label.
    job.delete_requests = None
    try:
        await db.commit()
    except Exception:
        # The live-version unique index racing a dialog-created twin between the check and
        # the commit; the query above catches the ordinary case and says it better.
        await db.rollback()
        raise HTTPException(
            status_code=409,
            detail=f"{job.character} v{job.version} already has a live run") from None
    await db.refresh(job)
    logger.info("retried training %s v%d from its snapshot (%d images in group 1)",
                job.character, job.version, len(job.dataset_images or []))
    return job
