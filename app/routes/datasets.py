"""Named, taggable training datasets (wanly-console#453).

WHAT THIS REPLACES. Before it, "these 27 images are p@y v2's training set" could not be written
down. The only groupings were an S3 folder prefix and a per-user favourites list, so every
training run meant re-selecting images by hand and a v2 meant doing it again from memory.

Uploads land in one S3 prefix per dataset, so a dataset is ALSO browsable as a folder in the
Image Repo -- nothing new to learn, and the images stay ordinary images. But the dataset is the
ordered LIST, not the folder listing: it survives an image being moved, and it fixes the order
the trainer stages them in, which is what the captions pair against.
"""
import asyncio
import base64
import logging
import uuid
from datetime import datetime, timezone
from typing import Literal

import random

import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Query, UploadFile
from sqlalchemy import cast, func, or_, select
from sqlalchemy.dialects.postgresql import JSONB, JSONPATH
from sqlalchemy.ext.asyncio import AsyncSession

from app import s3
from app.auth import get_current_user, verify_api_key_or_bearer
from app.config import settings
from app.database import async_session, get_db, release_connection
from app.enums import JobStatus, SegmentStatus, TrainingStatus
from app import clips
from app.joycaption import TRAINING_CAPTION, TRAINING_MOTION_CAPTION, CaptionError
from app.models import Dataset, Job, LtxCharacter, Segment, TrainingJob, TrainingRunDataset, User
from app import run_datasets
from app.ltx_stack import LTX_STACK
from app.regularization import (
    REG_FRAMES, REG_PRIORITY_BASE, reg_prompt, reg_recipe, reg_size, reg_tag,
)
from app.schemas.datasets import (
    DatasetCaptionEdit, DatasetCaptionStatus, DatasetCaptionsRun, DatasetClone, DatasetCreate,
    DatasetLock, DatasetRegularize, DatasetRegularizeStatus, DatasetResponse, DatasetScore, DatasetScores,
    DatasetTrainedBy, DatasetUpdate,
)
from app.seeds import new_seed
from app.tag_filter import tag_clause

logger = logging.getLogger(__name__)
router = APIRouter()

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


#: Every dataset's objects live under this top-level prefix, and the Image Repo does not list
#: it (see `list_folders`). Datasets used to land in a `dataset-<name>/` folder that showed up
#: in the repo beside the generation folders, which made it look as if the two were connected
#: (wanly-console#464). They are not: a dataset is training input, the repo is render input.
DATASETS_PREFIX = "datasets"


def _prefix(ds_id: uuid.UUID) -> str:
    """The S3 folder a dataset's uploads go into.

    Keyed by id, not by name, so a rename is just a rename -- the name was becoming part of
    every image's key forever, which is why renaming was never offered.
    """
    return f"{DATASETS_PREFIX}/{ds_id}"


def _prune_annotations(ds: Dataset, also_drop: set[str] | None = None) -> None:
    """Drop captions and scores for URIs that are no longer in the set (#352).

    Both are keyed by URI, so an entry for a removed or cropped-away image is not harmful on
    its own -- but it is a caption nobody can see or edit, and a crop that reuses a filename
    would inherit a stranger's. `also_drop` names URIs that are still in the set but whose
    CONTENT changed (a re-upload over the same key), whose old caption and score describe a
    different picture.

    Reassigned, never mutated: JSONB does not see an in-place change (see add_images).
    """
    keep = set(ds.images) - (also_drop or set())
    captions = {u: c for u, c in (ds.captions or {}).items() if u in keep}
    scores = {u: v for u, v in (ds.scores or {}).items() if u in keep}
    if captions != (ds.captions or {}):
        ds.captions = captions
    if scores != (ds.scores or {}):
        ds.scores = scores


async def _validate_ownership(db: AsyncSession, kind: str | None, character: str | None,
                              reg_class: str | None) -> tuple[str | None, str | None, str | None]:
    """The (kind, character, reg_class) a set may be stored with, or a 422 saying why not.

    The rules are the design, not validation for its own sake. A character set belongs to a
    REGISTERED solo character, because the training route reads the owner's trigger from
    the registry; a composition set belongs to a pair name, which must not also be a solo
    character (the pair would then publish over that person's row, the bug #352 removes);
    a regularization pool belongs to nobody and says which class word it stands in for.
    """
    if kind is None:
        # Unassigned is a real state, and it is ALL unassigned -- an owner without a kind
        # would be read by nothing and mislead whoever looks.
        return None, None, None
    if kind == "regularization":
        if character:
            raise HTTPException(status_code=422,
                                detail="a regularization pool belongs to no character")
        if not reg_class:
            raise HTTPException(status_code=422,
                                detail="a regularization pool needs its class: woman or man")
        return kind, None, reg_class
    if reg_class:
        raise HTTPException(status_code=422, detail="only a regularization pool has a class")
    if not character:
        raise HTTPException(status_code=422,
                            detail=f"a {kind} set needs an owner (a character name)")
    # CASE-INSENSITIVE, and stored as the registry spells it. "david" and "David" are one
    # person to whoever typed them; two spellings of one owner would split its sets between
    # a name the training route finds and one it does not.
    row = (await db.execute(select(LtxCharacter).where(
        func.lower(LtxCharacter.name) == character.lower()))).scalars().first()
    if row is not None:
        character = row.name
    if kind == "character":
        if row is None:
            raise HTTPException(status_code=422,
                                detail=f"{character!r} is not a registered character — "
                                       f"register it first, with its trigger and gender")
        if (row.kind or "solo") != "solo":
            raise HTTPException(status_code=422,
                                detail=f"{character!r} is a pair; a pair owns a composition "
                                       f"set, not a character set")
    else:  # composition
        if row is not None and (row.kind or "solo") == "solo":
            raise HTTPException(status_code=422,
                                detail=f"{character!r} is a solo character; a composition "
                                       f"set is owned by a PAIR name (e.g. DavidKelly-2026)")
    return kind, character, None


#: A run in any other state -- queued, claimed, running, completed -- means a LoRA came, or
#: is coming, of the set. A failed or cancelled run produced nothing, so it is not listed as
#: having trained the set (its files are still protected: see run_datasets.trained_uris).
_NO_LORA = (TrainingStatus.FAILED, TrainingStatus.CANCELLED)


async def _trained_by(db: AsyncSession,
                      ds_id: uuid.UUID | None = None) -> dict[str, list[DatasetTrainedBy]]:
    """{dataset id: every run with a LoRA that trained on it}, oldest first.

    INFORMATION, NOT A LOCK (#420). A set used to lock once it trained (#356), so the way to a
    v2 was a clone -- "Joana v1..v4". Now the RUN is the record (training_run_datasets, #422)
    and the set stays the subject's living set; this only says which runs it fed.

    Read from two places, because runs from before the link table have no rows until the
    backfill (#424): each run's recorded provenance -- group 0's `config.dataset.id` and every
    `identities[].dataset.id` -- and the link table's `dataset_id`. ONE QUERY each, for one set
    or for all of them, so the list endpoint is not N+1.
    """
    group0 = TrainingJob.config["dataset"]["id"].astext
    others = func.jsonb_path_query_array(
        TrainingJob.identities, cast("$[*].dataset.id", JSONPATH), type_=JSONB)
    arch = TrainingJob.config["arch"].astext
    q = (select(TrainingJob.id, TrainingJob.character, TrainingJob.version, TrainingJob.status,
                TrainingJob.created_at, arch, group0, others)
         .where(TrainingJob.status.not_in(list(_NO_LORA)))
         .order_by(TrainingJob.created_at.asc(), TrainingJob.id.asc()))
    if ds_id is not None:
        q = q.where(or_(group0 == str(ds_id),
                        TrainingJob.identities.contains([{"dataset": {"id": str(ds_id)}}]),
                        TrainingJob.id.in_(select(TrainingRunDataset.training_job_id)
                                           .where(TrainingRunDataset.dataset_id == ds_id))))
    rows = (await db.execute(q)).all()
    links: dict[str, set[str]] = {}
    lq = select(TrainingRunDataset.training_job_id, TrainingRunDataset.dataset_id).where(
        TrainingRunDataset.dataset_id.is_not(None))
    if ds_id is not None:
        lq = lq.where(TrainingRunDataset.dataset_id == ds_id)
    for job_id, linked in (await db.execute(lq)).all():
        links.setdefault(str(job_id), set()).add(str(linked))
    out: dict[str, list[DatasetTrainedBy]] = {}
    for job_id, character, version, status, created_at, run_arch, g0, rest in rows:
        run = DatasetTrainedBy(job_id=str(job_id), character=character, version=version,
                               status=status, arch=run_arch or "ltx", created_at=created_at)
        # A set used by two groups of one run (a joint run's member set that is also its
        # composition set, say) is still one run.
        for used in ({g0, *(rest or [])} | links.get(str(job_id), set())) - {None}:
            out.setdefault(str(used), []).append(run)
    return out


async def _runs_of(db: AsyncSession, ds: Dataset) -> list[DatasetTrainedBy]:
    """Every run with a LoRA that trained on this one set, oldest first."""
    return (await _trained_by(db, ds.id)).get(str(ds.id), [])


async def _used_in(db: AsyncSession) -> dict[str, list[DatasetTrainedBy]]:
    """{uri: the runs that trained on it} for every image any run used (#422)."""
    return {u: [DatasetTrainedBy(**r) for r in runs]
            for u, runs in (await run_datasets.used_in(db)).items()}


def _is_locked(ds: Dataset) -> bool:
    """Read-only: locked by hand (#358, optional) or archived (#419). Never by training."""
    return ds.locked_at is not None or ds.archived_at is not None


def _respond(ds: Dataset, trained_by: list[DatasetTrainedBy] | None,
             used_in: dict[str, list[DatasetTrainedBy]] | None = None) -> DatasetResponse:
    """The wire shape of a set, with its runs, per-image badges and lock filled in."""
    out = DatasetResponse.model_validate(ds)
    out.trained_by = list(trained_by or [])
    if used_in:
        mine = set(ds.images or [])
        out.used_in = {u: runs for u, runs in used_in.items() if u in mine}
    out.locked = _is_locked(ds)
    return out


async def _respond_one(db: AsyncSession, ds: Dataset) -> DatasetResponse:
    return _respond(ds, await _runs_of(db, ds), await _used_in(db))


def _lock_detail(ds: Dataset) -> str:
    """Why the set is read-only: archived, or locked by hand and why."""
    if ds.archived_at is not None:
        return (f"{ds.name!r} is archived (folded into its subject's set) — "
                f"unarchive it to make changes")
    reason = f" ({ds.locked_reason})" if ds.locked_reason else ""
    return f"{ds.name!r} is locked by hand{reason} — unlock it to make changes"


async def _refuse_if_locked(db: AsyncSession, ds: Dataset) -> None:
    """409 if the set is read-only: locked by hand (#358) or archived (#419).

    Called BEFORE any work or write: a refused crop must not have uploaded its faces first.
    Training no longer locks a set (#420): what a run trained on is the run's own record
    (training_run_datasets), and the files it names are never deleted or overwritten (#421).
    """
    if _is_locked(ds):
        raise HTTPException(status_code=409, detail=_lock_detail(ds))


@router.post("/datasets", response_model=DatasetResponse, status_code=201)
async def create_dataset(
    body: DatasetCreate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    dupe = (await db.execute(select(Dataset).where(Dataset.name == body.name))).scalar_one_or_none()
    if dupe:
        raise HTTPException(status_code=409, detail=f"a dataset called {body.name!r} already exists")
    kind, character, reg_class = await _validate_ownership(
        db, body.kind, body.character, body.reg_class)
    ds_id = uuid.uuid4()
    ds = Dataset(id=ds_id, user_id=user.id, name=body.name, tags=body.tags, notes=body.notes,
                 images=[], prefix=_prefix(ds_id), kind=kind, character=character,
                 reg_class=reg_class, captions={}, scores={})
    db.add(ds)
    await db.commit()
    await db.refresh(ds)
    # A new id: no run can have recorded it yet.
    return _respond(ds, None)


@router.get("/datasets", response_model=list[DatasetResponse],
            dependencies=[Depends(verify_api_key_or_bearer)])
async def list_datasets(
    #: Archived version sets (#419) are hidden unless asked for: they are history, folded
    #: into their subject's living set by the backfill.
    include_archived: bool = Query(False),
    db: AsyncSession = Depends(get_db),
):
    q = select(Dataset).order_by(Dataset.updated_at.desc())
    if not include_archived:
        q = q.where(Dataset.archived_at.is_(None))
    rows = (await db.execute(q)).scalars().all()
    # Two queries for the whole page, not one per set: every locking run is read once, and
    # each set's unlock (#363) is applied to its own runs here, in Python.
    trained_by = await _trained_by(db)
    used = await _used_in(db)
    return [_respond(ds, trained_by.get(str(ds.id)), used) for ds in rows]


@router.get("/datasets/{dataset_id}", response_model=DatasetResponse,
            dependencies=[Depends(verify_api_key_or_bearer)])
async def get_dataset(dataset_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    ds = await db.get(Dataset, dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return await _respond_one(db, ds)


@router.patch("/datasets/{dataset_id}", response_model=DatasetResponse)
async def update_dataset(
    dataset_id: uuid.UUID,
    body: DatasetUpdate,
    _user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    ds = await db.get(Dataset, dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")
    # A READ-ONLY SET (locked by hand, or archived) keeps its list. Refused before any field
    # is touched, so a PATCH that also renames is refused whole rather than half-applied. The
    # SAME list is not a change: a form that sends everything back with a new name must work.
    # A trained set is NOT read-only (#420): its runs recorded what they trained on.
    if (_is_locked(ds) and body.images is not None
            and list(body.images) != list(ds.images)):
        raise HTTPException(status_code=409, detail=_lock_detail(ds))
    # A CLIP ARRIVES ONLY THROUGH UPLOAD (#411), which normalizes it. One added by URI -- from
    # the Image Repo, say -- would be whatever fps, size and length it was rendered at, with its
    # audio, and could be too short for the trainer to cut a single window from. A clip already
    # under datasets/ came through an upload (another set's, or a clone), so it may move.
    if body.images is not None:
        foreign = [u for u in body.images if clips.is_clip(u) and u not in ds.images
                   and f"/{DATASETS_PREFIX}/" not in u]
        if foreign:
            raise HTTPException(
                status_code=422,
                detail=f"{len(foreign)} clip(s) can't be added from the repo — upload clips "
                       f"with Add images or clips, so they are converted for training")
    if body.name is not None and body.name != ds.name:
        dupe = (await db.execute(
            select(Dataset).where(Dataset.name == body.name, Dataset.id != ds.id)
        )).scalar_one_or_none()
        if dupe:
            raise HTTPException(status_code=409,
                                detail=f"a dataset called {body.name!r} already exists")
        ds.name = body.name
        # The prefix does not follow a rename: it is keyed by id, and moving objects that a
        # finished training job's dataset_images point at would break that job's record.
    if body.tags is not None:
        ds.tags = body.tags
    if body.notes is not None:
        ds.notes = body.notes
    anchor_before = ds.anchor_uri
    if body.images is not None:
        ds.images = body.images
        # An anchor that was just removed would silently score everything against nothing.
        if ds.anchor_uri and ds.anchor_uri not in body.images:
            ds.anchor_uri = None
        _prune_annotations(ds)
    if body.anchor_uri is not None:
        # "" clears it. None means the field was not sent, which must not clear anything --
        # the same distinction every other field here makes.
        ds.anchor_uri = body.anchor_uri or None
    if ds.anchor_uri != anchor_before:
        # Scores are likeness TO THE ANCHOR. Against a different face they are numbers about
        # somebody else, and the training route would pass or refuse the set on them.
        ds.scores = {}
    owner_fields = {"kind", "character", "reg_class"} & body.model_fields_set
    if owner_fields:
        new = {f: getattr(body, f) if f in owner_fields else getattr(ds, f)
               for f in ("kind", "character", "reg_class")}
        if new["kind"] is None and "kind" in owner_fields:
            new = {"kind": None, "character": None, "reg_class": None}
        kind, character, reg_class = await _validate_ownership(
            db, new["kind"], new["character"], new["reg_class"])
        if (kind, character, reg_class) != (ds.kind, ds.character, ds.reg_class):
            # A run snapshotted its captions and owner's trigger at creation, so this would
            # not corrupt it -- but the set would then say it is somebody else's while a LoRA
            # of the old owner came from it, and a publish would stamp a provenance that no
            # longer matches the set. Compared AFTER validation, so re-sending the owner as
            # the registry spells it (or in another case) is not a change. Nothing has been
            # committed yet, so the refusal leaves the row as it was.
            if _is_locked(ds):
                raise HTTPException(status_code=409, detail=_lock_detail(ds))
            ds.kind, ds.character, ds.reg_class = kind, character, reg_class
    await db.commit()
    await db.refresh(ds)
    return await _respond_one(db, ds)


@router.post("/datasets/{dataset_id}/images", response_model=DatasetResponse)
async def add_images(
    dataset_id: uuid.UUID,
    files: list[UploadFile] = File(...),
    _user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Upload images -- and video clips (#411) -- into a dataset. Many at once, because a
    dataset is 13-50 of them.

    A CLIP IS NORMALIZED BEFORE ANYTHING IS STORED (app/clips.py), and every clip in the batch
    is converted first: one that cannot train (too short, unreadable) refuses the whole upload
    with its name, rather than landing the rest and leaving the person to work out which file
    went missing.
    """
    ds = await db.get(Dataset, dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")
    # Before the uploads: a re-upload over an existing name overwrites the object in place,
    # which on a locked set would change a trained image under its LoRA's record.
    await _refuse_if_locked(db, ds)

    staged: list[tuple[str, bytes]] = []
    refused: list[str] = []
    for f in files:
        name = (f.filename or "image.jpg").rsplit("/", 1)[-1]
        ext = ("." + name.rsplit(".", 1)[1].lower()) if "." in name else ".jpg"
        if ext in clips.CLIP_SUFFIXES:
            try:
                data = await clips.normalize(await f.read(), ext)
            except clips.ClipError as e:
                refused.append(f"{name}: {e}")
                continue
            # Stored as what it now is, whatever it arrived as.
            staged.append((name.rsplit(".", 1)[0] + ".mp4", data))
            continue
        if ext not in IMAGE_SUFFIXES:
            # Skipped rather than fatal: one stray file in a folder drag-and-drop should not
            # reject the other forty-nine.
            logger.info("dataset %s: skipping %s (not an image or clip)", ds.name, name)
            continue
        staged.append((name, await f.read()))
    if refused:
        raise HTTPException(status_code=422,
                            detail="Nothing was added. " + "; ".join(refused))

    added: list[str] = []
    replaced: set[str] = set()
    # NEVER OVERWRITE WHAT A RUN TRAINED ON (#421). Re-uploading a filename used to replace
    # the object in place -- harmless while a trained set was frozen, but sets are living now
    # (#420), and the bytes under a run's recorded URI must stay the bytes it learned from. A
    # name that collides with a trained file lands under a fresh key instead, as a new image.
    trained = await run_datasets.trained_uris(db)
    for name, data in staged:
        key = f"{ds.prefix}/{name}"
        if f"s3://{settings.s3_images_bucket}/{key}" in trained:
            stem, dot, ext = name.rpartition(".")
            fresh = f"{stem}-{uuid.uuid4().hex[:8]}{dot}{ext}" if stem else \
                f"{name}-{uuid.uuid4().hex[:8]}"
            key = f"{ds.prefix}/{fresh}"
        uri = await asyncio.to_thread(
            s3.upload_bytes, data, key, settings.s3_images_bucket)
        if uri not in ds.images:
            added.append(uri)
        else:
            # Same key, NEW BYTES: the object was just overwritten, so its caption and score
            # describe a picture that no longer exists.
            replaced.add(uri)

    if replaced:
        _prune_annotations(ds, also_drop=replaced)
    if added:
        # Reassigned rather than appended in place: JSONB columns do not see a mutation of the
        # existing list, so `ds.images.append(...)` writes nothing and the upload silently
        # vanishes on the next read.
        ds.images = list(ds.images) + added
    if added or replaced:
        await db.commit()
        await db.refresh(ds)
    logger.info("dataset %s: added %d item(s), now %d", ds.name, len(added), len(ds.images))
    return _respond(ds, None)


@router.delete("/datasets/{dataset_id}", status_code=204)
async def delete_dataset(
    dataset_id: uuid.UUID,
    purge: bool = False,
    _user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete the dataset row. `purge=true` also deletes its images from S3.

    Not by default: a finished training job records the URIs it trained on, and destroying them
    turns a reproducible run into an unreproducible one.

    Refused while the set is locked by hand or archived. A set that trained CAN be deleted
    (#420): each run keeps its own record of what it trained on (training_run_datasets, whose
    dataset_id goes NULL but whose name and images stay), and a purge never deletes a file
    any run trained on (#421).

    A PURGE DELETES ONLY THIS SET'S OWN PREFIX, AND NOTHING ANOTHER SET STILL LISTS. A clone
    shares its source's URIs rather than copying the objects, so its images live under the
    SOURCE's prefix: purging the clone never reaches them (its own prefix holds only what was
    uploaded or cropped into the clone), and purging the source skips every object a clone
    -- or any other set -- still names. Removing an image from a set never deletes anything;
    it only edits the list.
    """
    ds = await db.get(Dataset, dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")
    await _refuse_if_locked(db, ds)
    if purge and ds.prefix:
        shared = {u for (images,) in (await db.execute(
            select(Dataset.images).where(Dataset.id != ds.id))).all() for u in images or []}
        # NOR ANYTHING A RUN TRAINED ON (#421), in any status: the run's snapshot names it,
        # and a retry trains exactly that snapshot (#423).
        shared |= await run_datasets.trained_uris(db)
        # Positional order is (prefix, bucket, ...) — the other call sites pass it that way.
        # Swapping them makes botocore reject the PREFIX as an invalid bucket name, which is
        # how every purge-delete 500'd: a legacy prefix like "dataset-test-faces" is not a
        # bucket. The trailing "/" keeps "datasets/<id>" from matching a longer sibling.
        await asyncio.to_thread(s3.delete_prefix_except, ds.prefix + "/",
                                settings.s3_images_bucket, shared)
    await db.delete(ds)
    await db.commit()


@router.post("/datasets/{dataset_id}/clone", response_model=DatasetResponse, status_code=201)
async def clone_dataset(
    dataset_id: uuid.UUID,
    body: DatasetClone,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """A copy of a set, under a new name.

    No longer the way to a v2 (#420): a set stays editable after it trains, because each run
    records what it trained on. Cloning is for a genuinely separate set -- an experiment that
    should not disturb the subject's living one. Reads the source; writes nothing to it.

    THE SAME URIs, NOT COPIES OF THE OBJECTS. The images are shared: the clone's list names
    the objects under the source's prefix. That is safe because nothing that edits a set
    deletes an object -- removal only edits the list, a crop writes a new batch under the
    set's OWN prefix and leaves the photographs where they were -- and a purge-delete keeps
    anything another set still lists (see delete_dataset). The clone gets its own prefix,
    keyed by its own id, for whatever is uploaded or cropped into it from now on.

    Everything that describes the images comes along -- captions and scores are keyed by URI,
    so they still pair; the anchor is one of the same URIs -- and so do kind, owner and class,
    so the clone trains as the same person's set without being re-assigned. The notes say
    where it came from, because the source's notes would otherwise read as the clone's own
    history.
    """
    src = await db.get(Dataset, dataset_id)
    if not src:
        raise HTTPException(status_code=404, detail="Dataset not found")
    dupe = (await db.execute(select(Dataset).where(Dataset.name == body.name))).scalar_one_or_none()
    if dupe:
        raise HTTPException(status_code=409, detail=f"a dataset called {body.name!r} already exists")
    ds_id = uuid.uuid4()
    # Lists and dicts copied, not shared: the ORM would otherwise hold the source's own
    # objects on the clone, and a later reassign on one must not be the other's too.
    ds = Dataset(id=ds_id, user_id=user.id, name=body.name, tags=src.tags,
                 notes=(f"Cloned from {src.name!r}. " + (src.notes or "")).strip(),
                 images=list(src.images or []), prefix=_prefix(ds_id),
                 anchor_uri=src.anchor_uri, kind=src.kind, character=src.character,
                 reg_class=src.reg_class, captions=dict(src.captions or {}),
                 scores=dict(src.scores or {}))
    db.add(ds)
    await db.commit()
    await db.refresh(ds)
    logger.info("dataset %s: cloned from %s (%d images, shared)", ds.name, src.name,
                len(ds.images))
    # A new id: no run can have recorded it, so a clone is always unlocked. Nor is a hand
    # lock copied (#358): the lock is on the source, and the clone exists to be changed.
    return _respond(ds, None)


@router.post("/datasets/{dataset_id}/lock", response_model=DatasetResponse)
async def lock_dataset(
    dataset_id: uuid.UUID,
    body: DatasetLock | None = None,
    _user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Lock a set by hand (#358): optional, off by default, for a set to keep as it is.

    Training no longer locks anything (#420). This sets `locked_at` and `locked_reason`; from
    then on the set refuses every change to its images and captions until /unlock. A set
    already locked comes back unchanged -- the first lock's time and reason stand, so a
    double-click cannot overwrite them.
    """
    ds = await db.get(Dataset, dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")
    if ds.locked_at is not None:
        logger.info("dataset %s: already locked by hand at %s; unchanged",
                    ds.name, ds.locked_at.isoformat())
        return await _respond_one(db, ds)
    ds.locked_at = datetime.now(timezone.utc)
    ds.locked_reason = body.reason if body else None
    await db.commit()
    await db.refresh(ds)
    logger.info("dataset %s: locked by hand (%s)", ds.name, ds.locked_reason or "no reason given")
    return await _respond_one(db, ds)


@router.post("/datasets/{dataset_id}/unlock", response_model=DatasetResponse)
async def unlock_dataset(
    dataset_id: uuid.UUID,
    _user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Lift a hand lock (#358): the set can change again. Records `unlocked_at` (#363).

    Training never locks a set (#420), so this is the only lock there is besides archiving.
    """
    ds = await db.get(Dataset, dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")
    was = _lock_detail(ds) if ds.locked_at is not None else "it was not locked"
    ds.unlocked_at = datetime.now(timezone.utc)
    ds.locked_at = None
    ds.locked_reason = None
    await db.commit()
    await db.refresh(ds)
    logger.info("dataset %s: unlocked; before: %s", ds.name, was)
    return await _respond_one(db, ds)


@router.post("/datasets/{dataset_id}/archive", response_model=DatasetResponse)
async def archive_dataset(
    dataset_id: uuid.UUID,
    _user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Archive a set (#419): hidden from lists and pickers, read-only, never deleted.

    What the backfill (#424) does to a version set it folded into its subject's living set.
    Runs keep linking to it -- its images are what they trained on. Idempotent.
    """
    ds = await db.get(Dataset, dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")
    if ds.archived_at is None:
        ds.archived_at = datetime.now(timezone.utc)
        await db.commit()
        await db.refresh(ds)
        logger.info("dataset %s: archived", ds.name)
    return await _respond_one(db, ds)


@router.post("/datasets/{dataset_id}/unarchive", response_model=DatasetResponse)
async def unarchive_dataset(
    dataset_id: uuid.UUID,
    _user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Bring an archived set back into lists and pickers, editable again. Idempotent."""
    ds = await db.get(Dataset, dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")
    if ds.archived_at is not None:
        ds.archived_at = None
        await db.commit()
        await db.refresh(ds)
        logger.info("dataset %s: unarchived", ds.name)
    return await _respond_one(db, ds)


@router.post("/datasets/{dataset_id}/crop", response_model=DatasetResponse)
async def crop_faces(
    dataset_id: uuid.UUID,
    largest_only: bool = False,
    # Query(None), not a bare default: a non-scalar annotation with no Query marker is a BODY
    # parameter, so the console's repeated `uris=` query keys were never read and every crop
    # ran on the whole set no matter what was selected (api#336).
    uris: list[str] | None = Query(None),
    save_as: bool = False,
    framing: Literal["face", "head_shoulders"] = "face",
    _user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Replace this dataset's images with the faces cropped out of them -- or, with `save_as`,
    keep the set and append the crops as new images instead.

    IN PLACE by default. It used to write a second dataset called "<name> faces" and leave this
    one as it was, on the argument that the photographs are the source of truth and a crop is
    derived. In use that meant every dataset came in pairs, the one you trained from was
    never the one you named, and the question "why did cropping make a new dataset?" was
    asked on the first try (wanly-console#464). The photographs are still in the bucket under
    the dataset's own prefix, so nothing is destroyed; the dataset simply IS the crops now,
    which is what anyone cropping wanted. `save_as=True` is the other half of that question:
    the set keeps its photographs and the crops join the end of it, so one run produces both.

    SELECTIVE. `uris` crops those and only those; absent means every image in the set. A
    25-image set that needed 5 faces re-cropped cropped all 25 to get them, and the 20 good
    crops came back different -- the output is not deterministic, so a re-run of the whole set
    silently replaced crops nobody asked to change. Selected URIs that are not in the set are
    ignored rather than fatal, the way a stale selection in a long-open dialog survives an
    image being removed in another tab.

    NO GATE, NO REFERENCE. Cropping used to take a reference dataset and drop faces that
    scored below the same-person floor against its mean. A mean over a set that still
    contains two people is a blend of both and separates neither, so the gate was already
    off unless a reference was named, and the dropdown for naming one was the single most
    confusing control on the page. Culling is done afterwards, against ONE anchor face the
    user picks, with the numbers on screen: POST /datasets/{id}/score.

    `largest_only` DEFAULTS FALSE: KEEP EVERY FACE. In a photo of two people "largest" is
    whoever stood closer to the camera. An unwanted crop is one click to remove; a missing
    one is a re-run.

    `framing` (#409): "face" is the tight square crop, as always; "head_shoulders" is a 4:5
    portrait from just above the hairline to the upper chest. A face-crop service that predates
    it ignores the field and sends face crops, so a head-and-shoulders request whose response
    does not echo the framing is refused rather than stored as the wrong thing.
    """
    ds = await db.get(Dataset, dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")
    # Before the downloads and the face-crop call: a refused crop must not have uploaded a
    # batch of faces nobody will ever see.
    await _refuse_if_locked(db, ds)
    if not settings.face_crop_url:
        raise HTTPException(
            status_code=503,
            detail="no face-crop service is configured (face_crop_url is empty)")
    if not ds.images:
        raise HTTPException(status_code=422, detail="this dataset has no images")

    # Stills only (#411): a clip is the face in motion, and cropping one frame out of it would
    # throw the motion away. Clips are left in the set untouched.
    targets = [u for u in (ds.images if uris is None else
                           [u for u in ds.images if u in set(uris)]) if not clips.is_clip(u)]
    if not targets:
        raise HTTPException(
            status_code=422,
            detail="none of the selected images are in this dataset — they may have been "
                   "removed, or they are clips, which are not cropped")

    # CONCURRENTLY. Fetched one at a time this was fourteen serial round trips to S3 before any
    # work started; they are independent and the wait is entirely network.
    blobs = await asyncio.gather(
        *(asyncio.to_thread(s3.download_bytes, u) for u in targets))
    payload = {
        "images": [base64.b64encode(b).decode() for b in blobs],
        "reference": [],
        "largest_only": largest_only,
        "framing": framing,
    }
    async with httpx.AsyncClient(timeout=settings.face_crop_timeout_s) as client:
        try:
            r = await client.post(f"{settings.face_crop_url.rstrip('/')}/crop", json=payload)
            r.raise_for_status()
        except httpx.HTTPError as e:
            raise HTTPException(status_code=503, detail=f"face-crop unreachable: {e}") from e
    result = r.json()
    if result.get("framing", "face") != framing:
        raise HTTPException(
            status_code=503,
            detail=f"the face-crop service does not support {framing!r} crops yet — it needs "
                   f"the worker image updated (wanly-gpu-docker#187)")
    faces = result["faces"]
    if not faces:
        raise HTTPException(status_code=422, detail="no faces were detected in any image")

    # THE EXTENSION FOLLOWS WHAT THE SERVICE ACTUALLY SENT. It returns JPEG now, capped at the
    # trainer's resolution ceiling, because full-resolution lossless PNG made an 80 MB response
    # that could not cross a home uplink inside the read timeout. `format` is absent on a
    # face-crop that predates that, and the old contract there was PNG -- so the two repos can
    # deploy in either order.
    ext = {"jpeg": "jpg"}.get(str(faces[0].get("format", "png")).lower(), "png")
    # A crop batch gets its own sub-folder, so — in either mode — cropping twice cannot
    # overwrite the first batch's files while a training job still records them. save_as does
    # NOT mean overwrite the originals with the crop: a URI a finished training job's
    # dataset_images points at must keep meaning the photograph it was created with.
    # Head-and-shoulders batches say so in the bucket (#409).
    kind = "portraits" if framing == "head_shoulders" else "faces"
    batch = f"{ds.prefix}/{kind}-{uuid.uuid4().hex[:6]}"

    # Uploaded concurrently, for the same reason the fetch is: independent, network-bound, and
    # serial round trips are the whole cost.
    async def put(i: int, f: dict) -> str:
        src = targets[f["source_index"]].rsplit("/", 1)[-1].rsplit(".", 1)[0]
        key = f"{batch}/{i:03d}_{src}_f{f['face_index']}.{ext}"
        return await asyncio.to_thread(
            s3.upload_bytes, base64.b64decode(f["png_b64"]), key, settings.s3_images_bucket)

    # The scope BEFORE the new URIs are computed: `uris` gets rebound below, and comparing
    # against the rebound value made every note claim "selected images" even when the whole
    # set was cropped — the lie that hid api#336 in the production data.
    scope = "every image" if uris is None else f"{len(targets)} selected images"
    crop_uris = list(await asyncio.gather(*(put(i, f) for i, f in enumerate(faces))))
    no_face = len(result.get("no_face", []))
    what = "head-and-shoulders crops" if framing == "head_shoulders" else "faces"
    note = (f"Cropped {len(faces)} {what} from {scope} "
            f"({'largest only' if largest_only else 'every face'})"
            + (f", {no_face} with none detected" if no_face else "")
            + ("; the set kept its photos and the crops joined it" if save_as else "") + ".")
    ds.notes = f"{ds.notes}\n{note}".strip() if ds.notes else note
    if save_as:
        # JSONB columns do not see an in-place mutation (see add_images); reassign.
        ds.images = list(ds.images) + crop_uris
        # The anchor is still a photograph in the set; scoring against it still works.
    else:
        replaced = set(targets)
        kept_others = [u for u in ds.images if u not in replaced]
        ds.images = kept_others + crop_uris
        # The anchor was one of the cropped photographs; the set is faces now — but an anchor
        # outside the selection is still what it was.
        if ds.anchor_uri in replaced:
            ds.anchor_uri = None
            # Scores were likeness to a photograph that is no longer in the set.
            ds.scores = {}
        # The photographs' captions and scores do not describe their crops.
        _prune_annotations(ds)
    await db.commit()
    await db.refresh(ds)
    logger.info("cropped %s (save_as=%s): %d faces from %d of %d photos",
                ds.name, save_as, len(crop_uris), len(targets),
                len(ds.images) - (len(crop_uris) if save_as else 0))
    return _respond(ds, None)


@router.post("/datasets/{dataset_id}/score", response_model=DatasetScores)
async def score_against_anchor(
    dataset_id: uuid.UUID,
    anchor_uri: str | None = None,
    _user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Score every image in a set against ONE image in it, nominated as the anchor.

    THE QUESTION THIS ANSWERS IS THE ONLY ONE WORTH ASKING: is this the same person as that.

    The reference machinery that came first scored against a whole dataset's MEAN, and that does
    not survive contact with a real set. A mean over images that still contain two people is a
    blend of both -- it separates neither, and whichever person is in the minority scores lower
    for no reason but being outnumbered. With no reference at all it scored against the crops'
    own mean, which shows they resemble each other and nothing else. One picked face has neither
    problem.

    IT RETURNS NUMBERS; IT DELETES NOTHING. Hand-culling failed twice in this project, but the
    failure was culling with no information, not culling by hand -- an unlabelled thumbnail of a
    stranger at a bad angle looks like a bad photo of the right person. A score beside each
    image fixes that, and leaves the judgement where it belongs. Removing is a separate,
    reversible click.

    The anchor is remembered on the dataset, so re-scoring after a cull compares against the
    same face rather than a moving target.
    """
    ds = await db.get(Dataset, dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")
    if not settings.face_crop_url:
        raise HTTPException(
            status_code=503,
            detail="no face-crop service is configured (face_crop_url is empty)")

    uri = anchor_uri or ds.anchor_uri
    if not uri:
        raise HTTPException(status_code=422, detail="pick an anchor image first")
    if uri not in ds.images:
        # Most likely it was removed since it was picked. Say which, rather than returning a
        # set of scores quietly measured against nothing.
        raise HTTPException(
            status_code=422,
            detail="the anchor is not in this dataset — it may have been removed; pick another")
    if clips.is_clip(uri):
        raise HTTPException(status_code=422, detail="the anchor must be a still image, not a clip")

    embeddings = await _embed_all(ds.images)
    anchor_vec = embeddings[ds.images.index(uri)][0]
    if not anchor_vec:
        raise HTTPException(
            status_code=422,
            detail="no face was detected in the anchor image — pick one that is a clear face")

    floor = settings.face_cos_floor
    # -2.0 is _cos's "one side had no embedding", which is not a low score but an absent one.
    # Surfaced as null so the console can say "no face" rather than rendering it as the worst
    # match in the set. A clip is the median over its frames (app/clips.clip_score).
    def score(vecs: list[list[float]]) -> float | None:
        cosines = [None if not e else round(_cos(e, anchor_vec), 4) for e in vecs]
        return cosines[0] if len(cosines) == 1 else clips.clip_score(cosines)

    scores = [DatasetScore(uri=u, cos=score(vecs), is_anchor=(u == uri))
              for u, vecs in zip(ds.images, embeddings)]

    # Remembered only once it has been shown to work on this set -- the anchor AND what the
    # set scored against it (#352). The training route reads the scores to refuse a
    # character set with somebody else's face in it; computing them and throwing them away
    # made that impossible without the face-crop box being up at the moment of training.
    ds.anchor_uri = uri
    ds.scores = {sc.uri: sc.cos for sc in scores}
    await db.commit()

    return DatasetScores(anchor_uri=uri, cos_floor=floor, scores=scores)


async def _embed_all(uris: list[str]) -> list[list[list[float]]]:
    """Each item's embeddings, parallel to `uris`: one for a still, one per sampled frame for
    a clip (#411). An empty embedding is "no face". One /embed call for the whole set."""
    blobs = await asyncio.gather(*(asyncio.to_thread(s3.download_bytes, u) for u in uris))
    per_item = [await clips.frames(b) if clips.is_clip(u) else [b] for u, b in zip(uris, blobs)]
    flat = [img for imgs in per_item for img in imgs]
    body = {"images": [base64.b64encode(b).decode() for b in flat]}
    async with httpx.AsyncClient(timeout=settings.face_crop_timeout_s) as client:
        r = await client.post(f"{settings.face_crop_url.rstrip('/')}/embed", json=body)
        r.raise_for_status()
    vecs = iter(r.json()["embeddings"])
    return [[next(vecs) for _ in imgs] for imgs in per_item]


def _cos(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return -2.0
    return sum(x * y for x, y in zip(a, b))


# ---------------------------------------------------------------------------------------
# Training captions (wanly-api#352)
#
# Every image used to train under the one caption "<trigger>, <gender>", so framing,
# clothing, lighting and background were absorbed into the trigger. Each image now gets its
# own caption BODY -- what varies in the frame, never who is in it (joycaption.
# TRAINING_CAPTION) -- stored per URI without the trigger. The training route adds the
# prefix when a run is created.
# ---------------------------------------------------------------------------------------

#: The captioning runs THIS PROCESS is doing, by dataset id: {"running": bool, "error": str}.
#: In memory, deliberately: a run is a loop in this process, so a restart ends it -- and the
#: progress that matters (which images have captions) is on the row, not here. After a
#: restart `running` reads false and the button can simply be pressed again; it fills only
#: what is missing.
_CAPTION_RUNS: dict[uuid.UUID, dict] = {}


def _caption_status(ds: Dataset) -> DatasetCaptionStatus:
    run = _CAPTION_RUNS.get(ds.id) or {}
    caps = ds.captions or {}
    return DatasetCaptionStatus(
        total=len(ds.images),
        captioned=sum(1 for u in ds.images if (caps.get(u) or "").strip()),
        running=bool(run.get("running")),
        error=run.get("error"),
    )


async def caption_dataset_images(db: AsyncSession, ds_id: uuid.UUID, overwrite: bool) -> int:
    """Caption a set's images one at a time, through the captioner's queue. Returns how many.

    SERIAL, through app/caption_queue.py, like bulk tagging's describe (images.
    describe_untagged): the captioner is one ollama slot, so parallel requests only wait
    inside it -- and inside an HTTP timeout. Taking a turn per image puts the set in line
    with every other caption, in a known order, and a toolbar can see it.

    INTERACTIVE, so the render gate applies: on the 3090 the captioner shares the card with
    the render stack, and a caption while it renders OOMs one of them. A refusal (the box is
    rendering, or in render mode, or the captioner is down) STOPS the loop and is recorded
    as the run's error, which the status endpoint shows. Nothing is lost: the captions
    already written are on the row, and the next run fills only what is missing.

    Each write re-reads the row first, so a caption edited by hand while the loop runs, or an
    image removed, is not overwritten from a stale copy.
    """
    from app.caption_queue import queue as caption_queue
    from app.routes.captions import caption_clip_sheet, caption_image_bytes

    ds = await db.get(Dataset, ds_id)
    if ds is None:
        return 0
    have = ds.captions or {}
    todo = [u for u in ds.images if overwrite or not (have.get(u) or "").strip()]
    done = 0
    for uri in todo:
        # No connection held while in line or while the captioner works (console#559): a
        # turn can be minutes away, and the transaction left open by the last read would sit
        # "idle in transaction" on a pooled connection the whole time.
        await release_connection(db)
        try:
            async with caption_queue.turn(uri, kind="dataset"):
                image = await asyncio.to_thread(s3.download_bytes, uri)
                if clips.is_clip(uri):
                    # What CHANGES across the clip, from a 2x2 sheet of its frames (#411).
                    text = await caption_clip_sheet(db, await clips.contact_sheet(image),
                                                    TRAINING_MOTION_CAPTION)
                else:
                    text, _ = await caption_image_bytes(db, image, instruction=TRAINING_CAPTION)
        except CaptionError as e:
            # Box-wide (busy, render mode, unreachable) and per-image refusals look alike
            # from here; stopping costs one press of the button, hammering costs the box.
            logger.warning("dataset %s: captioning stopped at %s: %s", ds.name, uri, e)
            (_CAPTION_RUNS.setdefault(ds_id, {}))["error"] = str(e)
            break
        except Exception:
            # An unreadable image is that image's problem, not the set's.
            logger.exception("dataset %s: could not caption %s; skipping", ds.name, uri)
            continue
        text = (text or "").strip()
        if not text:
            logger.warning("dataset %s: empty caption for %s; skipping", ds.name, uri)
            continue
        # FOR UPDATE, held only from here to the commit below -- milliseconds, never across the
        # captioner. It orders this write against DELETE /images dropping the same URI's
        # entries (console#559): without it, a delete landing between this read and the
        # commit is undone by writing back the dict read before it.
        await db.refresh(ds, with_for_update=True)
        if uri not in ds.images:
            continue
        # DELETED WHILE IT WAS BEING CAPTIONED. The delete keeps the set's membership (the
        # 409 dialog's "dead entry"), so the membership check above cannot see it -- and the
        # caption written here would be exactly the orphan the delete just dropped. One HEAD
        # per ~25 s caption; a HEAD that fails for any other reason skips the image too,
        # which the next run fills in.
        if await asyncio.to_thread(s3.head_object, uri) is None:
            logger.info("dataset %s: %s was deleted while captioning; not storing its caption",
                        ds.name, uri)
            continue
        # A lock by hand (#358) or an archive mid-loop stops it: the refresh above re-read
        # both. A run queued meanwhile does NOT (#420) -- it snapshotted its own captions,
        # so a caption written now is simply the living set's next one.
        if _is_locked(ds):
            logger.warning("dataset %s: captioning stopped at %s: %s",
                           ds.name, uri, _lock_detail(ds))
            (_CAPTION_RUNS.setdefault(ds_id, {}))["error"] = _lock_detail(ds)
            break
        ds.captions = {**(ds.captions or {}), uri: text}
        await db.commit()
        done += 1
    logger.info("dataset %s: captioned %d of %d", ds.name, done, len(todo))
    return done


async def caption_dataset_job(ds_id: uuid.UUID, overwrite: bool) -> None:
    """BackgroundTasks entry: its own session, because the request's is closed by now."""
    async with async_session() as db:
        try:
            await caption_dataset_images(db, ds_id, overwrite)
        except Exception as e:
            logger.exception("captioning dataset %s failed", ds_id)
            (_CAPTION_RUNS.setdefault(ds_id, {}))["error"] = f"{type(e).__name__}: {e}"
        finally:
            (_CAPTION_RUNS.setdefault(ds_id, {}))["running"] = False


@router.post("/datasets/{dataset_id}/captions", response_model=DatasetCaptionStatus,
             status_code=202)
async def caption_dataset(
    dataset_id: uuid.UUID,
    background_tasks: BackgroundTasks,
    body: DatasetCaptionsRun | None = None,
    _user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Caption every image in the set that has no caption yet (all of them with `overwrite`).

    Returns at once with the progress so far; poll GET .../captions/status. A second press
    while a run is going returns that run's progress rather than starting another -- two
    loops would caption the same images twice, and the second would win.
    """
    ds = await db.get(Dataset, dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")
    await _refuse_if_locked(db, ds)
    if not ds.images:
        raise HTTPException(status_code=422, detail="this dataset has no images")
    run = _CAPTION_RUNS.get(ds.id)
    if run and run.get("running"):
        return _caption_status(ds)
    _CAPTION_RUNS[ds.id] = {"running": True, "error": None}
    background_tasks.add_task(caption_dataset_job, ds.id, bool(body and body.overwrite))
    return _caption_status(ds)


@router.get("/datasets/{dataset_id}/captions/status", response_model=DatasetCaptionStatus,
            dependencies=[Depends(verify_api_key_or_bearer)])
async def caption_dataset_status(dataset_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    ds = await db.get(Dataset, dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return _caption_status(ds)


@router.patch("/datasets/{dataset_id}/captions", response_model=DatasetResponse)
async def edit_dataset_caption(
    dataset_id: uuid.UUID,
    body: DatasetCaptionEdit,
    _user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Replace one image's caption body by hand. A blank one deletes it.

    The body only -- no trigger, no gender. They are added when a run is created, from the
    registry; typed in here they would appear twice in the final caption.
    """
    ds = await db.get(Dataset, dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")
    await _refuse_if_locked(db, ds)
    if body.uri not in ds.images:
        raise HTTPException(status_code=422,
                            detail="that image is not in this dataset — it may have been removed")
    text = " ".join(body.caption.split())
    caps = dict(ds.captions or {})
    if text:
        caps[body.uri] = text
    else:
        caps.pop(body.uri, None)
    ds.captions = caps
    await db.commit()
    await db.refresh(ds)
    return _respond(ds, None)


# ---------------------------------------------------------------------------------------
# Regularization pools (wanly-api#352) -- see app/regularization.py for why they exist and
# why they are rendered from the base model.
#
# HOW A RENDER BECOMES AN IMAGE IN THE POOL. Each render is an ordinary Job with one
# segment 0, no start image and no character LoRA, claimed and rendered by the normal
# segment path; the daemon uploads its last frame as it does for every LTX segment. Nothing
# in that path knows about pools. The status endpoint COLLECTS: every completed segment of a
# job tagged for this pool has its last frame copied into the dataset's own prefix, and the
# job is archived, which is the mark that it has been collected. Polling is the console's
# job anyway (it shows progress), so the collection costs no new hook in the segment
# report path -- and a copy that fails is simply retried on the next poll.
# ---------------------------------------------------------------------------------------


@router.post("/datasets/{dataset_id}/regularize", response_model=DatasetRegularizeStatus,
             status_code=202)
async def regularize_dataset(
    dataset_id: uuid.UUID,
    body: DatasetRegularize,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Queue `count` text-to-video renders of generic people for a regularization pool."""
    ds = await db.get(Dataset, dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")
    # A pool that trained a LoRA is part of that LoRA's record like any other set; growing
    # it is adding images. Clone it and generate into the clone.
    await _refuse_if_locked(db, ds)
    if ds.kind != "regularization" or not ds.reg_class:
        raise HTTPException(
            status_code=422,
            detail="only a regularization dataset (kind=regularization, with a class) can be "
                   "generated into")
    fps = LTX_STACK["frame_rate"]
    tag = reg_tag(ds.id)
    rng = random.Random()
    # Behind everything, and behind any pool already queued, in the order asked for.
    top = (await db.execute(select(func.coalesce(func.max(Job.priority), REG_PRIORITY_BASE - 1))
                            .where(Job.priority >= REG_PRIORITY_BASE))).scalar_one()
    for i in range(body.count):
        w, h = reg_size(rng)
        job = Job(user_id=user.id, name=f"Reg {ds.reg_class} · {ds.name}",
                  width=w, height=h, fps=fps, seed=new_seed(), priority=top + 1 + i,
                  tags=f"regularization, {tag}", status=JobStatus.PENDING)
        db.add(job)
        await db.flush()
        db.add(Segment(job_id=job.id, index=0, prompt=reg_prompt(ds.reg_class, rng),
                       duration_seconds=REG_FRAMES / fps, speed=1.0, start_image=None,
                       ltx_recipe=reg_recipe(ds.reg_class), auto_finalize=False))
    await db.commit()
    logger.info("dataset %s: queued %d %s regularization render(s)",
                ds.name, body.count, ds.reg_class)
    return await _collect_regularization(db, ds)


async def _collect_regularization(db: AsyncSession, ds: Dataset) -> DatasetRegularizeStatus:
    """Copy every finished, uncollected render's last frame into the pool; count the rest.

    Idempotent: the destination key is the segment's id, the URI is added only if absent,
    and a collected job is archived so it is never looked at again -- including after a
    human removes its frame from the pool as a bad render, which must not bring it back.
    """
    jobs = (await db.execute(select(Job).where(tag_clause(Job.tags, reg_tag(ds.id))))
            ).scalars().all()
    # A READ-ONLY POOL IS NOT GROWN: locked by hand (#358) or archived (#419). Its late
    # renders stay uncollected, counted as still running, until it is unlocked. A pool that
    # trained is not read-only (#420): the run recorded which frames it used.
    locked = _is_locked(ds)
    if locked:
        logger.info("dataset %s: not collecting regularization frames: %s",
                    ds.name, _lock_detail(ds))
    requested = len(jobs)
    pending = rendering = failed = collected = collected_now = 0
    for job in jobs:
        if job.status == JobStatus.ARCHIVED:
            collected += 1
            continue
        seg = (await db.execute(
            select(Segment).where(Segment.job_id == job.id, Segment.discarded.is_(False))
            .order_by(Segment.index.asc()).limit(1))).scalar_one_or_none()
        if seg is None or seg.status == SegmentStatus.FAILED:
            failed += 1
            continue
        if seg.status == SegmentStatus.PENDING:
            pending += 1
            continue
        if seg.status != SegmentStatus.COMPLETED or not seg.last_frame_path or locked:
            rendering += 1
            continue
        ext = seg.last_frame_path.rsplit(".", 1)[-1].lower() if "." in seg.last_frame_path \
            else "png"
        key = f"{ds.prefix}/reg/{seg.id}.{ext}"
        try:
            data = await asyncio.to_thread(s3.download_bytes, seg.last_frame_path)
            uri = await asyncio.to_thread(s3.upload_bytes, data, key, settings.s3_images_bucket)
        except Exception as e:
            logger.warning("dataset %s: could not collect %s (%s); will retry on the next poll",
                           ds.name, seg.last_frame_path, e)
            rendering += 1
            continue
        if uri not in ds.images:
            # JSONB: reassigned, never appended (see add_images).
            ds.images = list(ds.images) + [uri]
        job.status = JobStatus.ARCHIVED
        collected += 1
        collected_now += 1
    if collected_now:
        await db.commit()
        await db.refresh(ds)
        logger.info("dataset %s: collected %d regularization frame(s)", ds.name, collected_now)
    return DatasetRegularizeStatus(requested=requested, done=collected, failed=failed,
                                   running=pending + rendering, pending=pending,
                                   collected_now=collected_now, images=len(ds.images))


@router.get("/datasets/{dataset_id}/regularize/status", response_model=DatasetRegularizeStatus)
async def regularize_status(
    dataset_id: uuid.UUID,
    _user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Where the pool's renders are -- and, as a side effect, collect the finished ones.

    A GET that writes is unusual; it is the least invasive place for it (see the section
    comment), and it is idempotent, which is what GET actually promises.
    """
    ds = await db.get(Dataset, dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return await _collect_regularization(db, ds)
