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
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app import s3
from app.auth import get_current_user, verify_api_key, verify_api_key_or_bearer
from app.config import settings
from app.database import get_db
from app.enums import TRAINING_TERMINAL, TrainingStatus, WorkerKind, worker_can
from app.models import Dataset, LtxCharacter, TrainingJob, User, Worker
from app.character_registry import identity_phrase
from app.schemas.training import (
    MIN_DATASET_IMAGES, TrainingClaimResponse, TrainingCreate, TrainingNotes,
    TrainingPreflight, TrainingProgress, TrainingResponse,
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
        config={**RECIPE_DEFAULTS,
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
                "reg_ratio": REG_RATIO,
                "allow_no_composition": body.allow_no_composition,
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
        #: Provenance, snapshot at creation: a rename must not rewrite what trained.
        "dataset": g.provenance(),
    }


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
    if (job.config or {}).get("mode") in ("solo", "pair"):
        await _publish_registered(db, job)
        return
    # ---- A PRE-#352 JOB, published exactly as it always was. Retrying an old failed run
    # must not change what it publishes to.
    #
    # THE STEM, NOT THE FILENAME. Every character row stores `pay_v2_e05`, the console's Use
    # button writes the stem, and the character editor says "without .safetensors" -- the
    # first auto-published character stored `david_v1_final.safetensors` and the page could
    # not tell it was in use.
    basename = job.output_lora_path.rsplit("/", 1)[-1].removesuffix(".safetensors")
    existing = (await db.execute(
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


async def _publish_registered(db: AsyncSession, job: TrainingJob) -> None:
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
    basename = _lora_stem(job)
    existing = (await db.execute(
        select(LtxCharacter).where(LtxCharacter.name == job.character)
    )).scalar_one_or_none()
    stamp = {"char_lora": basename, "trained_from": _trained_from(job),
             "base_checkpoint": cfg.get("base_checkpoint")}
    if cfg.get("mode") == "pair":
        phrases = [identity_phrase(job.trigger, cfg.get("gender"))]
        phrases += [identity_phrase(g.get("trigger"), g.get("gender"))
                    for g in (job.identities or []) if g.get("kind") == "identity"]
        stamp.update(kind="pair", members=list(cfg.get("members") or []),
                     trigger=" and ".join(p for p in phrases if p), gender=None)
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


def _provenance_dataset_id(prov: dict) -> uuid.UUID | None:
    """The recorded dataset id as a real UUID — the column is as_uuid, so asyncpg refuses
    the string the JSONB holds. An id that does not parse is treated as absent: the
    snapshot trains rather than a retry dying on a provenance field."""
    raw = prov.get("id")
    if not raw:
        return None
    try:
        return uuid.UUID(str(raw))
    except ValueError:
        return None


def _reresolve_images(label: str, images: list, ds: Dataset | None) -> tuple[list, dict | None]:
    """Re-read one group's images from its dataset, when it has one and it still exists.

    Returns (images, refreshed provenance-or-None). The dataset WINS over the snapshot:
    a run that died on dead S3 objects is exactly the run whose images someone has just
    fixed, and re-queuing the frozen list would die on the same 404s. A dataset that has
    been deleted is not a retry-blocker — the snapshot may still train, and it is the
    caller's judgment whether to — so provenance is kept as recorded then.
    """
    if ds is None:
        return list(images), None
    fresh = list(ds.images)
    if len(fresh) < MIN_DATASET_IMAGES:
        raise HTTPException(
            status_code=422,
            detail=f"{label}: dataset {ds.name!r} now has {len(fresh)} images — at least "
                   f"{MIN_DATASET_IMAGES} are needed")
    if len(set(fresh)) != len(fresh):
        raise HTTPException(
            status_code=422, detail=f"{label}: dataset {ds.name!r} contains duplicates")
    provenance = {"id": str(ds.id), "name": ds.name, "count": len(fresh)}
    return fresh, provenance


def _reresolve_captioned(label: str, images: list, captions: list,
                         ds: Dataset | None) -> tuple[list, list, dict | None]:
    """`_reresolve_images` for a group with a caption snapshot (#352).

    The dataset decides WHICH images (in its current order); the snapshot decides every
    caption. An image the snapshot has no caption for is dropped -- it was never previewed,
    and a retry is not the place to caption it. What is left must still clear the floor.
    """
    snap = dict(zip(images, captions))
    if ds is None:
        return list(images), list(captions), None
    fresh = [u for u in ds.images if u in snap]
    if len(fresh) < MIN_DATASET_IMAGES:
        raise HTTPException(
            status_code=422,
            detail=f"{label}: dataset {ds.name!r} now has {len(fresh)} of this run's captioned "
                   f"images — at least {MIN_DATASET_IMAGES} are needed. Create a new run to "
                   f"train on its current images and captions.")
    if len(set(fresh)) != len(fresh):
        raise HTTPException(
            status_code=422, detail=f"{label}: dataset {ds.name!r} contains duplicates")
    provenance = {"id": str(ds.id), "name": ds.name, "count": len(fresh)}
    return fresh, [snap[u] for u in fresh], provenance


@router.post("/training/{job_id}/retry", response_model=TrainingResponse)
async def retry_training_job(
    job_id: uuid.UUID,
    _user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Queue a failed run again, in place, after re-reading its datasets (api#342).

    THE POINT IS THE RE-READ. A job snapshots its images at creation, which is right for a
    claim (a worker must never look its data up for itself) and wrong for a retry: the run
    failed, the human fixed the images, and the first retry died on the same dead URIs.
    So each group's images come from its dataset now, when the dataset still exists —
    the same provenance the console showed, trusted back for the second opinion it cannot
    give (it cannot know what changed on S3). A group trained from an ad-hoc URI list has
    no dataset to consult, and its snapshot stands.

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

    # ALL re-resolution happens BEFORE any mutation: a 422 must leave the row exactly as
    # it was — still failed, still showing its error — not half-rewritten.
    config = dict(job.config or {})
    # A #352 job carries per-image captions. Its retry re-reads each set's IMAGES (the
    # point of retry) but takes every caption from the job's own SNAPSHOT, never the
    # dataset: the run is re-queued as what was previewed and approved, and an image that
    # arrived since has no approved caption, so it is left out rather than guessed at.
    snapshotted = config.get("captions") is not None
    group0 = dict(config.get("dataset") or {})
    ds_id = _provenance_dataset_id(group0)
    ds = await db.get(Dataset, ds_id) if ds_id else None
    if snapshotted:
        images, captions, refreshed = _reresolve_captioned(
            "group 1", job.dataset_images, config["captions"], ds)
        config["captions"] = captions
        config["caption"] = captions[0] if captions else None
    else:
        images, refreshed = _reresolve_images("group 1", job.dataset_images, ds)
    if refreshed:
        config["dataset"] = {**group0, **refreshed}

    groups = [dict(g) for g in (job.identities or [])]
    for i, g in enumerate(groups):
        prov = dict(g.get("dataset") or {})
        g_id = _provenance_dataset_id(prov)
        gds = await db.get(Dataset, g_id) if g_id else None
        if g.get("captions") is not None:
            g_images, g_caps, g_refreshed = _reresolve_captioned(
                f"group {i + 2}", g.get("images") or [], g["captions"], gds)
            g["captions"] = g_caps
            g["caption"] = g_caps[0] if g_caps else None
        else:
            g_images, g_refreshed = _reresolve_images(
                f"identity {i + 2}", g.get("images") or [], gds)
        g["images"] = g_images
        if g_refreshed:
            g["dataset"] = {**prov, **g_refreshed}

    thumbnail = ds.anchor_uri if ds and ds.anchor_uri else job.thumbnail_uri

    twin = (await db.execute(
        select(TrainingJob).where(
            TrainingJob.character == job.character,
            TrainingJob.version == job.version,
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
    job.dataset_images = images
    job.config = config
    job.identities = groups or None
    job.thumbnail_uri = thumbnail
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
    logger.info("retried training %s v%d (%d images after re-reading its datasets)",
                job.character, job.version, len(job.dataset_images))
    return job
