import asyncio
import logging
import re
import uuid
from typing import Literal, Optional
from datetime import datetime, timezone

from fastapi import APIRouter, BackgroundTasks, Depends, Form, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, Response
from sqlalchemy import and_, func, not_, or_, select, text, true, update
from sqlalchemy import delete as sa_delete
from sqlalchemy.dialects.postgresql import array as pg_array
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import get_current_user, verify_api_key_or_bearer, verify_api_key_or_token
from app import caption_tickets
from app.config import settings
from app.routes.datasets import DATASETS_PREFIX
from app.database import async_session, get_db, release_connection
from app.joycaption import CaptionError, CaptionerBusy
from app.enums import TRAINING_TERMINAL
from app.models import Dataset, Favorite, ImageMeta, Job, LtxCharacter, Segment, TrainingJob, User
from app.routes.captions import caption_image_pair, caption_image_scene
from app.schemas.images import (BulkImageTagsUpdate, CaptionLane, CaptionQueueEntry,
                                CaptionQueueStatus,
                                CaptionTicket, CaptionTryRequest, CaptionTryResponse,
                                ImageSceneRequest, ImageSceneResponse, ImageTagsUpdate)
from app.tag_filter import like_escape, normalise_tag
from app.tag_filter import tag_clause as _tag_clause
from app.s3 import (
    delete_object,
    delete_prefix,
    download_bytes,
    get_folder_info,
    head_object,
    list_common_prefixes,
    list_objects,
    move_object,
    upload_bytes,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["images"])

_FOLDER_NAME_RE = re.compile(r"^[a-zA-Z0-9 _-]+$")


async def _meta_by_path(db: AsyncSession, paths: list[str]) -> dict[str, dict]:
    """The image_meta fields a listing shows, keyed by path.

    One helper rather than a copy of the same select in each of the four listings. There
    were three copies when this only had to fetch tags, and adding the scene description
    would have made four — which is how one listing quietly ends up returning a field the
    others do not.

    Columns, not entities: a listing needs three values per row and never mutates one, and
    the whole point of the folder view is that it stays cheap over a few thousand objects.
    """
    if not paths:
        return {}
    rows = (await db.execute(
        select(ImageMeta.path, ImageMeta.tags,
               ImageMeta.scene_description, ImageMeta.scene_described_at,
               ImageMeta.motion_description, ImageMeta.motion_described_at)
        .where(ImageMeta.path.in_(paths))
    )).all()
    return {
        row[0]: {
            "tags": row[1] or None,
            "scene_description": row[2] or None,
            "scene_described_at": row[3],
            "motion_description": row[4] or None,
            "motion_described_at": row[5],
        }
        for row in rows
    }


_NO_META = {"tags": None, "scene_description": None, "scene_described_at": None,
            "motion_description": None, "motion_described_at": None}


def _meta_fields(meta) -> dict:
    """The image_meta half of a listing row. Every listing returns the same shape.

    Takes either the mapping _meta_by_path builds or an ImageMeta itself — search already
    holds the entity, and making it re-fetch the same row as columns would be a second
    query for data it has. None is an image with no row.

    An image with no row is not a different shape from one with a row: it is the same
    fields, all empty. Returning fewer keys is how a client ends up with `undefined` where
    it expected null.
    """
    if meta is None:
        return dict(_NO_META)
    if isinstance(meta, ImageMeta):
        return {
            "tags": meta.tags or None,
            "scene_description": meta.scene_description or None,
            "scene_described_at": meta.scene_described_at,
        }
    return dict(meta)

# Every column that can hold an s3:// path a user could delete through this router.
# Job.identity_reference_image is deliberately absent: the daemon writes it into the *jobs*
# bucket, so it can never be the target of DELETE /images.
_JOB_IMAGE_COLUMNS = (Job.starting_image, Job.lynx_subject_image)
_SEGMENT_IMAGE_COLUMNS = (Segment.start_image,)


async def find_image_references(db: AsyncSession, paths: list[str]) -> dict[str, dict[str, list[str]]]:
    """Map each still-referenced path to the jobs and segments holding it.

    Deleting a referenced image fails silently: nothing breaks until a worker claims the
    segment weeks later and S3 returns 404, which costs a pickup and shows up as a red
    segment nowhere near the cause. That makes the check wider than it first looks:

      - segment-level refs count, not just Job.starting_image. A continuation's start frame
        lives on Segment.start_image, which was invisible to the old listing query.
      - ARCHIVED jobs count. Archiving hides a job from the UI; it does not stop its segments
        being re-run, so an archived job's images are still live references.
      - no user filter, since a reference from anyone's job 404s the worker just the same.
    """
    if not paths:
        return {}

    wanted = set(paths)
    refs: dict[str, dict[str, list[str]]] = {}

    def _hold(path: str | None, kind: str, holder_id) -> None:
        if not path or path not in wanted:
            return
        entry = refs.setdefault(path, {"job_ids": [], "segment_ids": [],
                                       "training_ids": [], "dataset_ids": []})
        if str(holder_id) not in entry[kind]:
            entry[kind].append(str(holder_id))

    job_rows = await db.execute(
        select(Job.id, *_JOB_IMAGE_COLUMNS).where(
            or_(*[col.in_(paths) for col in _JOB_IMAGE_COLUMNS])
        )
    )
    for row in job_rows.all():
        for value in row[1:]:
            _hold(value, "job_ids", row[0])

    seg_rows = await db.execute(
        select(Segment.id, *_SEGMENT_IMAGE_COLUMNS).where(
            or_(*[col.in_(paths) for col in _SEGMENT_IMAGE_COLUMNS])
        )
    )
    for row in seg_rows.all():
        for value in row[1:]:
            _hold(value, "segment_ids", row[0])

    # TRAINING DATASETS COUNT TOO (wanly-api#274). A queued training job holds its dataset as a
    # JSONB list of s3:// URIs, and deleting one of those images is the same silent failure as
    # deleting a start frame: nothing breaks until the trainer claims the job, fetches a 404,
    # and fails a run that was queued days earlier for a reason nowhere near the cause.
    #
    # Matched in Python rather than SQL because the paths live inside a JSON array; the row
    # count here is tiny (one per training run, ever) so a containment query would be more
    # machinery than the problem deserves.
    train_rows = await db.execute(
        select(TrainingJob.id, TrainingJob.dataset_images)
        .where(TrainingJob.status.not_in(list(TRAINING_TERMINAL)))
    )
    for job_id, images in train_rows.all():
        for value in images or []:
            _hold(value, "training_ids", job_id)

    # DATASET MEMBERSHIP COUNTS TOO (wanly-api#305). A photograph in a dataset that could be
    # deleted from the repo silently vanished from every set holding it: the set keeps a dead
    # URI, the count changes with no name, and training fetches a 404. Same Python-side JSONB
    # matching as the training check above -- one row per dataset ever, so a containment query
    # would be machinery the problem does not deserve.
    #
    # Unlike an in-flight training run, membership is ordinary state: someone rebuilding a set
    # legitimately wants the originals gone afterwards. That is what DELETE /datasets/{id}
    # (purge) and removing the image FROM the dataset are for -- this gate says which sets
    # hold it, and force=true stays the escape, named (wanly-api#156).
    ds_rows = await db.execute(select(Dataset.id, Dataset.images))
    for ds_id, images in ds_rows.all():
        for value in images or []:
            _hold(value, "dataset_ids", ds_id)

    return refs


def _job_referenced_paths(refs: dict[str, dict[str, list[str]]]) -> set[str]:
    """Paths a job or segment points at — the only ones that earn the Image Repo's green dot.

    The dot means "Used in a job" (wanly-console#168): a video exists or is coming. But the
    delete gate needs *every* way an image can be live, so `find_image_references` was widened
    to training runs (#274) and dataset membership (#305). The listing endpoints fed `in_use`
    off that whole return value, so every member of any dataset lit up with a dot and no video
    behind it (issue #389). The gate keeps the wide answer; the dot takes only the half that
    means output exists.
    """
    return {p for p, r in refs.items() if r["job_ids"] or r["segment_ids"]}


@router.post("/images/upload", dependencies=[Depends(verify_api_key_or_bearer)])
async def upload_image(
    file: UploadFile,
    filename: str | None = None,
    folder: str | None = Form(None),
):
    data = await file.read()
    if not filename:
        ext = ""
        if file.filename and "." in file.filename:
            ext = "." + file.filename.rsplit(".", 1)[1]
        else:
            ext = ".png"
        filename = f"{uuid.uuid4().hex}{ext}"
    if folder:
        prefix = folder.strip()
    else:
        prefix = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    key = f"{prefix}/{filename}"
    bucket = settings.s3_images_bucket
    uri = await asyncio.to_thread(upload_bytes, data, key, bucket)
    return {"path": uri}


@router.post("/images/folders", dependencies=[Depends(get_current_user)])
async def create_folder(body: dict):
    name = body.get("name", "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="Folder name is required")
    if len(name) > 100:
        raise HTTPException(status_code=400, detail="Folder name too long (max 100)")
    if not _FOLDER_NAME_RE.match(name):
        raise HTTPException(
            status_code=400,
            detail="Folder name may only contain letters, numbers, spaces, dashes, and underscores",
        )
    bucket = settings.s3_images_bucket
    marker_key = f"{name}/.folder"
    await asyncio.to_thread(upload_bytes, b"", marker_key, bucket)
    return {"name": name}


@router.get("/images/folders", dependencies=[Depends(verify_api_key_or_bearer)])
async def list_folders():
    """List folders in the images bucket, sorted by creation date newest first."""
    bucket = settings.s3_images_bucket
    prefixes = await asyncio.to_thread(list_common_prefixes, bucket)
    # Datasets share the bucket but are not repo folders: they are training input, and
    # listing them here made the two look connected (wanly-console#464).
    prefixes = [p for p in prefixes if p.rstrip("/") != DATASETS_PREFIX]

    async def _folder_info(prefix: str) -> dict:
        name = prefix.rstrip("/")
        info = await asyncio.to_thread(get_folder_info, bucket, prefix)
        thumbnail = f"s3://{bucket}/{info["key"]}" if info and info.get("key") else None
        created_at = info["created_at"] if info else None
        return {"name": name, "thumbnail": thumbnail, "created_at": created_at}

    folders = await asyncio.gather(*[_folder_info(p) for p in prefixes])
    # Sort by created_at descending (newest first); folders with no date go last
    folders.sort(key=lambda f: f["created_at"] or "", reverse=True)
    return list(folders)


@router.get("/images/folder/{date}", dependencies=[Depends(get_current_user)])
async def list_folder_images(
    date: str,
    db: AsyncSession = Depends(get_db),
):
    """List images in a date folder, with in_use flag indicating if used by any job."""
    bucket = settings.s3_images_bucket
    prefix = f"{date}/"
    objects = await asyncio.to_thread(list_objects, bucket, prefix)

    paths = [f"s3://{bucket}/{obj['Key']}" for obj in objects if not obj["Key"].endswith("/.folder")]
    in_use_set: set[str] = set()
    meta_map: dict[str, dict] = {}
    if paths:
        # Same helper the delete endpoint gates on, narrowed to job/segment refs: the dot
        # means "Used in a job", and dataset membership must not light it (issue #389).
        in_use_set = _job_referenced_paths(await find_image_references(db, paths))
        meta_map = await _meta_by_path(db, paths)

    return [
        {
            "key": obj["Key"],
            "path": f"s3://{bucket}/{obj['Key']}",
            "filename": obj["Key"].split("/", 1)[1] if "/" in obj["Key"] else obj["Key"],
            "size": obj["Size"],
            "last_modified": obj["LastModified"],
            "in_use": f"s3://{bucket}/{obj['Key']}" in in_use_set,
            **_meta_fields(meta_map.get(f"s3://{bucket}/{obj['Key']}")),
        }
        for obj in objects
        if not obj["Key"].endswith("/.folder")
    ]


def _is_dataset_path(uri: str) -> bool:
    """Is this s3:// URI inside the datasets prefix? The key is everything after the bucket."""
    return _is_dataset_key(uri.split("/", 3)[-1])


@router.get("/images/favorites", dependencies=[Depends(get_current_user)])
async def list_favorite_images(
    user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return all favorited images across all folders with metadata.

    A dataset URI that was favorited is filtered rather than rendered: datasets are training
    input (wanly-console#464's split), the search and folder listing already exclude them, and
    one leaked view makes the Image Repo look connected to the datasets. A stale favorite on a
    dataset image drops out here rather than surfacing as a grid entry; the Favorite row itself
    stays — this endpoint renders, it does not curate.
    """
    result = await db.execute(
        select(Favorite.item_ref)
        .where(Favorite.user_id == user.id, Favorite.item_type == "image")
        .order_by(Favorite.created_at.desc())
    )
    refs = [row[0] for row in result.all()]
    refs = [r for r in refs if not _is_dataset_path(r)]

    async def _meta(uri: str) -> dict | None:
        obj = await asyncio.to_thread(head_object, uri)
        if not obj:
            return None
        key = obj["Key"]
        return {
            "key": key,
            "path": uri,
            "filename": key.split("/", 1)[1] if "/" in key else key,
            "size": obj["Size"],
            "last_modified": obj["LastModified"],
        }

    items = await asyncio.gather(*[_meta(ref) for ref in refs])

    uris = [item["path"] for item in items if item is not None]
    meta_map = await _meta_by_path(db, uris)
    for item in items:
        if item is not None:
            item.update(_meta_fields(meta_map.get(item["path"])))

    return [item for item in items if item is not None]


def _is_dataset_key(key: str) -> bool:
    """Is this S3 key inside the datasets prefix? DATASETS_PREFIX with no trailing slash is
    the folder boundary question: "datasets-extra/" is a repo folder, "datasets/" is not.
    """
    return key.startswith(f"{DATASETS_PREFIX}/")


@router.get("/images/untagged", dependencies=[Depends(get_current_user)])
async def list_untagged_images(
    db: AsyncSession = Depends(get_db),
):
    """Return all images across all folders that have no tags — for tagging triage.

    "Untagged" means no image_meta row at all, or a row with empty/whitespace tags.
    Since never-tagged images have no row, this is a cross-folder scan minus the
    set of paths with non-empty tags. Dataset staging images are skipped entirely:
    they are training input (wanly-console#464's split), tags do not apply to them,
    and showing them here made the Image Repo look connected to the datasets.
    """
    bucket = settings.s3_images_bucket
    prefixes = await asyncio.to_thread(list_common_prefixes, bucket)
    object_lists = await asyncio.gather(
        *[asyncio.to_thread(list_objects, bucket, prefix) for prefix in prefixes]
    )
    objects = [
        obj
        for sublist in object_lists
        for obj in sublist
        if not obj["Key"].endswith("/.folder") and not _is_dataset_key(obj["Key"])
    ]
    paths = [f"s3://{bucket}/{obj['Key']}" for obj in objects]

    tagged: set[str] = set()
    in_use_set: set[str] = set()
    meta_map: dict[str, dict] = {}
    if paths:
        meta_map = await _meta_by_path(db, paths)
        tagged = {p for p, m in meta_map.items() if (m["tags"] or "").strip()}

        in_use_set = _job_referenced_paths(await find_image_references(db, paths))

    # An untagged image can still carry a description — describing one is offered in the
    # modal whether or not it has tags — so the row is read here rather than assumed empty.
    untagged = [
        {
            "key": obj["Key"],
            "path": f"s3://{bucket}/{obj['Key']}",
            "filename": obj["Key"].split("/", 1)[1] if "/" in obj["Key"] else obj["Key"],
            "size": obj["Size"],
            "last_modified": obj["LastModified"],
            "in_use": f"s3://{bucket}/{obj['Key']}" in in_use_set,
            **_meta_fields(meta_map.get(f"s3://{bucket}/{obj['Key']}")),
        }
        for obj in objects
        if f"s3://{bucket}/{obj['Key']}" not in tagged
    ]
    untagged.sort(key=lambda x: x["last_modified"], reverse=True)
    return untagged


@router.post("/images/move", dependencies=[Depends(get_current_user)])
async def move_images(body: dict, db: AsyncSession = Depends(get_db)):
    """Move one or more images to a target folder (S3 copy + delete).

    The image_meta row moves WITH the object. It did not before, so a move silently dropped
    an image's tags — survivable when a tag is five seconds of typing, not when the row also
    holds a scene description that cost GPU time and cannot be reproduced word for word
    (console#414). The path is that row's primary key, so this is a rename, not a copy.
    """
    keys: list[str] = body.get("keys", [])
    target_folder: str = body.get("target_folder", "").strip()
    if not keys:
        raise HTTPException(status_code=400, detail="No keys provided")
    if not target_folder:
        raise HTTPException(status_code=400, detail="target_folder is required")
    bucket = settings.s3_images_bucket

    async def _move_one(src_key: str) -> tuple[str, str]:
        filename = src_key.split("/", 1)[1] if "/" in src_key else src_key
        dst_key = f"{target_folder}/{filename}"
        await asyncio.to_thread(move_object, bucket, src_key, dst_key)
        return src_key, dst_key

    moved = await asyncio.gather(*[_move_one(k) for k in keys])

    # After the objects, deliberately. If a move fails the metadata must still describe
    # where the image actually is, and a row pointing at the old key is right in that case.
    for src_key, dst_key in moved:
        src = f"s3://{bucket}/{src_key}"
        dst = f"s3://{bucket}/{dst_key}"
        if src == dst:
            continue
        meta = await db.get(ImageMeta, src)
        if meta is None:
            continue
        # The path is the primary key, so this is a delete plus an insert. A row already at
        # the destination — the same filename moved back into a folder it came from — is
        # overwritten by the one that travelled with the object.
        existing = await db.get(ImageMeta, dst)
        if existing is not None:
            await db.delete(existing)
            await db.flush()
        db.add(ImageMeta(
            path=dst,
            tags=meta.tags,
            scene_description=meta.scene_description,
            scene_instruction=meta.scene_instruction,
            scene_described_at=meta.scene_described_at,
            motion_description=meta.motion_description,
            motion_instruction=meta.motion_instruction,
            motion_described_at=meta.motion_described_at,
        ))
        await db.delete(meta)
    # A CHARACTER'S SHEET OR FACE REF MOVES WITH THE IMAGE (wanly-console#581). The row names
    # the image by URI, and a claim presigns that URI -- left behind, every render of the
    # character would fail its reference download (or, with no LoRA, have no identity at all).
    for src_key, dst_key in moved:
        src = f"s3://{bucket}/{src_key}"
        dst = f"s3://{bucket}/{dst_key}"
        if src == dst:
            continue
        for col in (LtxCharacter.sheet_uri, LtxCharacter.face_ref_uri):
            await db.execute(update(LtxCharacter).where(col == src).values({col.key: dst}))
    await db.commit()

    return {"moved": len(moved)}


#: How long a delete waits for a row another writer holds before it gives up (console#559).
#: Every writer of these rows holds them for milliseconds -- none across a captioner call --
#: so a wait this long means something is wrong, and the person should hear so rather than
#: watch a spinner.
DELETE_LOCK_TIMEOUT = "5s"


def _is_lock_timeout(e: DBAPIError) -> bool:
    """Postgres's lock_not_available (55P03), which is what lock_timeout raises."""
    orig = getattr(e, "orig", None)
    return "55P03" in (getattr(orig, "sqlstate", None), getattr(orig, "pgcode", None))


async def _forget_images(db: AsyncSession, paths: list[str]) -> None:
    """Drop what the database says about images that are being deleted. Does not commit.

    An image's own caption and tags live on its image_meta row; a dataset holding it keeps a
    training caption and a likeness score keyed by its URI. Deleting the file used to leave
    all of them behind: a row describing nothing, and dataset entries nobody can see or edit
    -- which a later upload reusing the filename would silently inherit. Removing an image
    FROM a dataset, and cropping, already prune the dataset half (_prune_annotations); this
    is the same rule for deleting it outright (console#559).

    Membership is NOT touched. A force delete of a dataset image is a decision the 409 dialog
    has already spelled out, dead entry and all, and rewriting a locked set's image list from
    here would be a lock bypass by another name.

    FAILS FAST. lock_timeout bounds any wait for a dataset row, and a timeout is a 503 that
    says what happened. The caption loop holds its row only between re-reading it and
    committing a caption -- never across the captioner call -- so this should not wait at all;
    if it ever does, a clear refusal is the contract, not a hang.
    """
    if not paths:
        return
    try:
        await db.execute(text(f"SET LOCAL lock_timeout = '{DELETE_LOCK_TIMEOUT}'"))
        await db.execute(sa_delete(ImageMeta).where(ImageMeta.path.in_(paths)))
        keys = pg_array(paths)
        holders = (await db.execute(
            select(Dataset)
            .where(or_(Dataset.captions.has_any(keys), Dataset.scores.has_any(keys)))
            .with_for_update()
        )).scalars().all()
    except DBAPIError as e:
        if _is_lock_timeout(e):
            await db.rollback()
            raise HTTPException(
                status_code=503,
                detail="A dataset holding this image is being written to right now; "
                       "nothing was deleted. Try again in a moment.",
            ) from e
        raise
    gone = set(paths)
    for ds in holders:
        # Reassigned, never mutated: JSONB does not see an in-place change.
        ds.captions = {u: c for u, c in (ds.captions or {}).items() if u not in gone}
        ds.scores = {u: v for u, v in (ds.scores or {}).items() if u not in gone}
    if holders:
        logger.info("delete: dropped caption/score entries for %d image(s) from dataset(s) %s",
                    len(gone), ", ".join(ds.name for ds in holders))


@router.delete("/images", dependencies=[Depends(get_current_user)])
async def delete_image(
    path: str = Query(...),
    force: bool = Query(False),
    db: AsyncSession = Depends(get_db),
):
    """Delete a single image by S3 URI, refusing while a job or segment still points at it.

    force=true skips the check for when the deletion is genuinely intended and the resulting
    dangling reference is accepted.

    The image's caption, tags and dataset caption/score entries go with it (_forget_images).
    Nothing here waits on captioning: no caption-queue turn, no captioner call, and no row
    the caption loop holds across one (console#559).
    """
    bucket = settings.s3_images_bucket
    if not path.startswith(f"s3://{bucket}/"):
        raise HTTPException(status_code=400, detail="Path must be in the images bucket")
    if not force:
        refs = await find_image_references(db, [path])
        if path in refs:
            raise HTTPException(
                status_code=409,
                detail={
                    "message": "Image is still referenced; pass force=true to delete anyway",
                    "path": path,
                    "job_ids": refs[path]["job_ids"],
                    "segment_ids": refs[path]["segment_ids"],
                    "dataset_ids": refs[path]["dataset_ids"],
                },
            )
    await _forget_images(db, [path])
    # The object last, the commit after it: a failed S3 delete rolls the forgetting back
    # (the session closes uncommitted), so an image that is still there keeps its captions.
    await asyncio.to_thread(delete_object, path)
    await db.commit()
    return {"ok": True}


@router.delete("/images/folder")
async def delete_folder(
    name: str = Query(...),
    force: bool = Query(False),
    db: AsyncSession = Depends(get_db),
    user=Depends(get_current_user),
    _api_key: None = Depends(verify_api_key_or_bearer),
):
    """Delete an image folder, refusing while any image in it is still referenced.

    All items in the directory are deleted and are unrecoverable — that is what force
    confirms. The check is the same gate DELETE /images uses (jobs, segments, archived
    jobs, datasets), applied to every image in the folder at once: one reference check and
    one batch S3 delete beat N x per-image calls on a folder of hundreds. The 409 names
    every holding job/segment/dataset, so the warning the console pops can say exactly
    what will dangle.

    A query param, not a path segment: folder names are dates but the datasets check has
    to be reachable for the names that are not -- and a path param cannot span the "/" in
    "datasets/faces-abc" ({name} matches one segment only). DELETE /images takes its path
    the same way.

    The datasets/ prefix is not deletable here: dataset removal is the Datasets page's job,
    and a folder delete that could reach training input would make the repo split (wanly-
    console#464) lie. The empty prefix itself matches nothing.
    """
    bucket = settings.s3_images_bucket
    if not name or name.strip() != name or not name.strip():
        raise HTTPException(status_code=400, detail="Folder name is required")
    prefix = name.strip().rstrip("/") + "/"
    if _is_dataset_key(prefix):
        raise HTTPException(
            status_code=400,
            detail="Dataset folders are not deleted here — remove them from the Datasets page",
        )
    if prefix == "/":
        raise HTTPException(status_code=400, detail="Folder name is required")

    objects = await asyncio.to_thread(list_objects, bucket, prefix)
    paths = [f"s3://{bucket}/{obj['Key']}" for obj in objects]
    if not paths:
        raise HTTPException(status_code=404, detail=f"Folder '{name}' not found or empty")

    refs = await find_image_references(db, paths)
    if refs and not force:
        raise HTTPException(
            status_code=409,
            detail={
                "message": "Folder has images still referenced; pass force=true to delete anyway",
                "folder": name,
                "image_count": len(paths),
                "referenced_count": len(refs),
                "paths": {p: {"job_ids": r["job_ids"], "segment_ids": r["segment_ids"],
                              "dataset_ids": r["dataset_ids"]}
                          for p, r in refs.items()},
            },
        )

    await _forget_images(db, paths)
    deleted = await asyncio.to_thread(delete_prefix, prefix, bucket)
    await db.commit()
    return {"ok": True, "deleted": deleted, "folder": name.strip()}


@router.patch("/images/tags", dependencies=[Depends(get_current_user)])
async def update_image_tags(
    path: str = Query(...),
    body: ImageTagsUpdate = None,
    db: AsyncSession = Depends(get_db),
):
    """Update tags for an image by S3 URI."""
    bucket = settings.s3_images_bucket
    if not path.startswith(f"s3://{bucket}/"):
        raise HTTPException(status_code=400, detail="Path must be in the images bucket")

    result = await db.execute(select(ImageMeta).where(ImageMeta.path == path))
    meta = result.scalar_one_or_none()

    if body and body.tags:
        tags_val = body.tags.strip()
        if not tags_val:
            tags_val = None
    else:
        tags_val = None

    if meta:
        meta.tags = tags_val
        # Deleting the row on a cleared tag was fine when tags were all it held. A scene
        # description costs 2070 time to produce and cannot be regenerated identically, so
        # the row goes only when nothing is left on it at all (console#414).
        if meta.is_empty():
            await db.delete(meta)
    elif tags_val is not None:
        db.add(ImageMeta(path=path, tags=tags_val))

    await db.commit()
    return {"path": path, "tags": tags_val}


def _split_tags(blob: str | None) -> list[str]:
    """Split a stored tag blob into its trimmed members, dropping empties."""
    return [t.strip() for t in (blob or "").split(",") if t.strip()]


async def describe_untagged(db: AsyncSession, paths: list[str]) -> int:
    """Describe the SCENE of each of `paths` that has none yet, serially.

    Scene only (console#590): the motion paragraph is minutes of the motion captioner, and
    is made only when someone clicks Describe motion or a held job needs it -- tagging must
    not spend one. A saved motion paragraph is left as it is.

    The captioning half of bulk tagging (api#340): tagging an image is the moment someone
    decided it was worth keeping, which is the rule console#414 built single-image
    auto-describe on — but the bulk endpoint has no client loop to hang it on, and a
    client loop would die with the tab. So the description happens here, after the tag
    commit, one image at a time: each caption is two sequential Ollama calls on a GPU
    shared with the render stack, so parallelism buys nothing and OOMs something.

    The loop stops at the first captioner refusal (render busy, box down) and leaves the
    rest undescribed — visibly, in the lightbox. A later bulk-add re-queues them; retry
    with backoff across a ~30-minute render is not worth a queue table for one user.

    Returns the number described. The caller must have committed the tags already: a
    failure here must never take a tag write with it.
    """
    described = 0
    for path in paths:
        meta = await db.get(ImageMeta, path)
        if meta is None or (meta.scene_description or "").strip():
            continue  # row gone, or someone described it in the meantime
        if caption_tickets.active(path, caption_tickets.SCENE) is not None:
            continue  # a scene of it is already coming (console#564); a second would race it
        try:
            image = await asyncio.to_thread(download_bytes, path)
            scene, instruction = await caption_image_scene(db, image)
        except CaptionerBusy as e:
            logger.info("auto-describe batch stopping at %s: %s", path, e)
            break
        except CaptionError as e:
            # A box-wide refusal (unreachable) and a per-image refusal look the same from
            # here, and the cost of guessing wrong is one wasted caption per image. Stop
            # rather than hammer whatever is wrong.
            logger.warning("auto-describe stopping at %s: %s", path, e)
            break
        except Exception:
            # An unreadable image (deleted between commit and here) is that image's
            # problem, not the batch's.
            logger.exception("auto-describe could not read %s; skipping", path)
            continue
        if not (scene or "").strip():
            logger.warning("auto-describe got an empty caption for %s; skipping", path)
            continue
        caption_tickets.apply_scene(meta, scene, instruction)
        await db.commit()
        described += 1
    return described


async def describe_untagged_job(paths: list[str]) -> None:
    """BackgroundTasks entry: describe_untagged on its own session.

    Background tasks run after the response is sent and the request's session is closed,
    so this opens one of its own, the way stitch_video does.
    """
    async with async_session() as db:
        try:
            n = await describe_untagged(db, paths)
            logger.info("bulk-tag auto-describe: %d of %d described", n, len(paths))
        except Exception:
            logger.exception("bulk-tag auto-describe failed")


@router.post("/images/tags", dependencies=[Depends(get_current_user)])
async def bulk_update_image_tags(
    body: BulkImageTagsUpdate,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """Add or remove the same tags across many images at once (console#517).

    Server-side merge rather than a client loop over PATCH /images/tags: that endpoint
    replaces the whole blob with what the browser last saw, so N read-modify-writes race
    the lightbox and each other. Merging here, in one transaction, means dedupe runs
    against the row as it is now, not as it was when the list was fetched.

    Dedupe is normalise_tag (case- and space-insensitive), the same folding the search
    clause uses — adding "Kelly" to a row already holding "kelly " must not create a
    second chip for it.

    Remove keeps the row when it still holds a description (the console#414 rule): the
    row's primary key is the path and it carries GPU-produced captions, so "tags became
    empty" only deletes it when is_empty() agrees.

    All-or-nothing: a merged blob that would not fit the 500-char column refuses the whole
    request with the offending paths, rather than half-tagging the selection.
    """
    bucket = settings.s3_images_bucket
    bad = [p for p in body.paths if not p.startswith(f"s3://{bucket}/")]
    if bad:
        raise HTTPException(status_code=400, detail="Every path must be in the images bucket")

    wanted = _split_tags(body.tags)
    if not wanted:
        raise HTTPException(status_code=400, detail="No usable tags provided")
    # Dedupe the incoming set itself, first spelling wins: "Kelly, kelly" is one tag.
    seen: set[str] = set()
    new_tags: list[str] = []
    for t in wanted:
        key = normalise_tag(t)
        if key and key not in seen:
            seen.add(key)
            new_tags.append(t)
    if not new_tags:
        raise HTTPException(status_code=400, detail="No usable tags provided")

    metas = (await db.execute(
        select(ImageMeta).where(ImageMeta.path.in_(body.paths))
    )).scalars().all()
    meta_map = {m.path: m for m in metas}

    over_limit: list[str] = []
    planned: dict[str, str | None] = {}
    remove_keys = {normalise_tag(t) for t in new_tags}

    for path in body.paths:
        meta = meta_map.get(path)
        existing = _split_tags(meta.tags if meta else None)
        if body.mode == "add":
            have = {normalise_tag(t) for t in existing}
            merged = existing + [t for t in new_tags if normalise_tag(t) not in have]
        else:
            merged = [t for t in existing if normalise_tag(t) not in remove_keys]
        merged_val = ", ".join(merged) if merged else None
        if merged_val is not None and len(merged_val) > 500:
            over_limit.append(path)
            continue
        planned[path] = merged_val

    if over_limit:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Adding these tags would overflow the 500-character limit",
                "paths": over_limit,
            },
        )

    results = []
    to_describe: list[str] = []
    for path, merged_val in planned.items():
        meta = meta_map.get(path)
        if meta is None:
            # "No row" means "never tagged" to the untagged view, so a remove that changes
            # nothing must not create an empty row — only an add materialises one.
            if merged_val is not None:
                db.add(ImageMeta(path=path, tags=merged_val))
                results.append({"path": path, "tags": merged_val, "changed": True})
                to_describe.append(path)  # a new row has never been described
            else:
                results.append({"path": path, "tags": None, "changed": False})
            continue
        before = _split_tags(meta.tags)
        after = _split_tags(merged_val)
        # Checked before the merge writes: an add that moves the tags and has no scene
        # description yet is exactly what single-image auto-describe would have captioned
        # (console#414), so the batch captions it too (api#340). From the DB row, not the
        # client's cache — the console has no way to say what the captioner has done.
        if body.mode == "add" and before != after and not (meta.scene_description or "").strip():
            to_describe.append(path)
        meta.tags = merged_val
        if meta.is_empty():
            await db.delete(meta)
        results.append({"path": path, "tags": merged_val,
                        "changed": before != after})

    await db.commit()
    if to_describe:
        background_tasks.add_task(describe_untagged_job, to_describe)
    return {"results": results, "mode": body.mode, "describing": len(to_describe)}


# ---------------------------------------------------------------------------------------
# Scene descriptions (console#414)
#
# JoyCaption's description of a frame, stored on the image rather than re-derived per job.
# POST /captions/describe stays as it is: a stateless preview for bytes nobody has committed
# to. This is the other case — an image that lives in the repo, described once, by a person
# who is looking at it.
# ---------------------------------------------------------------------------------------


def _scene_response(path: str, meta: ImageMeta | None,
                    motion_error: str | None = None) -> ImageSceneResponse:
    description = (meta.scene_description if meta else None) or None
    motion = (meta.motion_description if meta else None) or None
    return ImageSceneResponse(
        path=path,
        scene_description=description,
        scene_instruction=(meta.scene_instruction if meta else None) or None,
        scene_described_at=meta.scene_described_at if meta else None,
        words=len(description.split()) if description else 0,
        motion_description=motion,
        motion_instruction=(meta.motion_instruction if meta else None) or None,
        motion_described_at=meta.motion_described_at if meta else None,
        motion_words=len(motion.split()) if motion else 0,
        motion_error=motion_error,
        caption=_ticket_response(path, caption_tickets.latest(path)),
        scene_caption=_ticket_response(path, caption_tickets.latest(path, caption_tickets.SCENE)),
        motion_caption=_ticket_response(path,
                                        caption_tickets.latest(path, caption_tickets.MOTION)),
        **_queue_fields(path),
    )


def _queue_fields(path: str) -> dict:
    """Where this image sits in the caption queue, for any response that mentions it.

    Read at response time rather than stored: a position recorded a moment ago is wrong as
    soon as anything ahead of it finishes. The image's own caption ticket when it has one
    (console#564) -- the same image can be in line twice (a dataset caption and a describe),
    and the ticket is the one that will land on this row.
    """
    from app.caption_queue import queue as caption_queue

    t = caption_tickets.active(path)
    if t is not None:
        v = caption_tickets.view(t)
        return {"queue_status": v["status"], "queue_position": v["position"],
                "queue_depth": v["depth"]}
    st = caption_queue.status(path)
    return {"queue_status": st["status"], "queue_position": st["position"],
            "queue_depth": st["depth"]}


def _ticket_response(path: str, t, joined: bool = False) -> CaptionTicket | None:
    if t is None:
        return None
    return CaptionTicket(path=path, joined=joined, **caption_tickets.view(t))


def _describable_buckets() -> tuple[str, ...]:
    """Where a frame this API will describe may live.

    The images bucket is the repo. The jobs bucket holds generated frames — a segment's last
    frame is the start frame of the one after it, and describing it is how a continuation
    shows the words before they are used (console#438). Both are ours; anything else is not
    a path this endpoint should be fetching.
    """
    return tuple(b for b in (settings.s3_images_bucket, settings.s3_jobs_bucket) if b)


def _require_known_bucket(path: str) -> None:
    if not any(path.startswith(f"s3://{b}/") for b in _describable_buckets()):
        raise HTTPException(
            status_code=400,
            detail="Path must be in the images bucket or the jobs bucket",
        )


def _requesters(ticket_id: str | None) -> list[dict]:
    """The held jobs a queued ticket is for, so the queue poll alone can say "Motion requested
    by job ..." (console#590)."""
    t = caption_tickets.get(ticket_id) if ticket_id else None
    return [dict(r) for r in t.requested_by] if t is not None else []


@router.get("/images/caption-queue", response_model=CaptionQueueStatus,
            dependencies=[Depends(verify_api_key_or_bearer)])
async def caption_queue_status():
    """How the captioner's queue looks, without naming an image.

    The per-image fields on /images/scene only help inside the modal of an image you are
    already describing. This is the one a toolbar can ask: it needs no path, so it is a
    single cheap poll that answers "is anything captioning, and how much is behind it"
    whatever page you are on.

    No database and no captioner call -- the queue is in this process.
    """
    from app import caption_queue as cq
    from app.joycaption import scene_status

    lanes = cq.lanes()
    return CaptionQueueStatus(
        depth=sum(q.depth() for _, q in lanes),
        waiting=sum(len(q.waiting_paths()) for _, q in lanes),
        running=cq.queue.running_path(),
        entries=[CaptionQueueEntry(path=e["path"], kind=e["kind"], status=e["status"],
                                   position=e["position"], ticket_id=e["token"], lane=name,
                                   requested_by=_requesters(e["token"]))
                 for name, q in lanes for e in q.entries()],
        lanes=[CaptionLane(name=name, depth=q.depth(), waiting=len(q.waiting_paths()),
                           running=q.running_path()) for name, q in lanes],
        scene_captioner=scene_status(),
        recent=[_ticket_response(t.path, t) for t in caption_tickets.recent()],
    )


@router.get("/images/scene", response_model=ImageSceneResponse,
            dependencies=[Depends(get_current_user)])
async def get_image_scene(path: str = Query(...), db: AsyncSession = Depends(get_db)):
    """This image's saved description, or nulls if it has never been described.

    Nulls rather than a 404: "no description yet" is an ordinary state of a perfectly good
    image, and the caller — the New Job modal — has to render it either way.
    """
    _require_known_bucket(path)
    return _scene_response(path, await db.get(ImageMeta, path))


def _describe_params(body: ImageSceneRequest | None) -> dict:
    body = body or ImageSceneRequest()
    return {k: v for k, v in body.model_dump(exclude={"halves"}).items() if v is not None}


def _halves(body: ImageSceneRequest | None) -> list[str]:
    """The halves asked for, scene first. None (an old client) means both."""
    asked = body.halves if body is not None and body.halves else list(caption_tickets.HALVES)
    return [h for h in caption_tickets.HALVES if h in asked]


def _request_halves(path: str, body: ImageSceneRequest | None):
    """One ticket per half asked for: [(ticket, joined)], scene first."""
    params = _describe_params(body)
    return [caption_tickets.request(path, half, origin="describe", params=params)
            for half in _halves(body)]


@router.post("/images/scene/describe", response_model=CaptionTicket, status_code=202,
             dependencies=[Depends(get_current_user)])
async def request_image_scene(
    path: str = Query(...),
    body: ImageSceneRequest | None = None,
    db: AsyncSession = Depends(get_db),
):
    """Describe this image in the background; answer at once with its caption ticket.

    console#564. The describe the console uses: the caption takes its turn in the queue in a
    task of its own, and the console polls GET /images/scene/status (or the whole queue,
    GET /images/caption-queue) and fills the words in when the ticket is done. Navigating
    away, closing the tab or a phone going to sleep loses nothing.

    PER HALF (console#590): `halves` is ["scene"], ["motion"] or both (the default, for old
    clients). Each half is a ticket of its own, in its own lane, and saving one never clears
    or overwrites the other -- so this one call is Describe, Describe motion, Redo scene,
    Redo motion and each half's Retry. Motion is grounded on the SAVED scene (the one in
    flight, when a scene is being made).

    ALWAYS regenerates the halves asked for -- unless a caption of that half is already
    queued or running, in which case this IS that caption (joined=true). Two captions of one
    half would write two different descriptions, and a held job might already be using the
    first (console#562).

    The answer is the first half's ticket, with every half's in `tickets`.
    """
    _require_known_bucket(path)
    # Nothing here waits, but the auth lookup opened a transaction; give the connection back.
    await release_connection(db)
    asked = _request_halves(path, body)
    t, joined = asked[0]
    out = _ticket_response(path, t, joined=joined)
    out.tickets = [_ticket_response(path, tt, joined=j) for tt, j in asked]
    return out


@router.get("/images/scene/status", response_model=CaptionTicket,
            dependencies=[Depends(verify_api_key_or_bearer)])
async def image_scene_status(path: str = Query(...),
                             half: Optional[Literal["scene", "motion"]] = Query(None)):
    """This image's caption ticket: queued (with position), running, done or failed.

    `half` picks the scene's or the motion's (console#590). Without it: whichever is in
    flight (the scene first), else the one that finished last. `scene` and `motion` carry
    both halves either way.

    status null: nothing in flight and nothing remembered -- either never asked for, or
    finished long enough ago (or before a restart) that the saved words are the answer.
    No database: tickets live in this process.
    """
    _require_known_bucket(path)
    t = caption_tickets.latest(path, half)
    out = (CaptionTicket(path=path, **caption_tickets.view(None)) if t is None
           else _ticket_response(path, t))
    out.scene = _ticket_response(path, caption_tickets.latest(path, caption_tickets.SCENE))
    out.motion = _ticket_response(path, caption_tickets.latest(path, caption_tickets.MOTION))
    return out


@router.get("/images/scene/tickets/{ticket_id}", response_model=CaptionTicket,
            dependencies=[Depends(verify_api_key_or_bearer)])
async def image_scene_ticket(ticket_id: str):
    """One caption ticket by id. 404 once it is forgotten (RESULT_TTL_S, or a restart)."""
    t = caption_tickets.get(ticket_id)
    if t is None:
        raise HTTPException(status_code=404, detail="no such caption ticket (finished long "
                                                    "ago, or the API restarted)")
    return _ticket_response(t.path, t)


@router.post("/images/scene", response_model=ImageSceneResponse,
             dependencies=[Depends(get_current_user)])
async def describe_image_scene(
    path: str = Query(...),
    body: ImageSceneRequest | None = None,
    db: AsyncSession = Depends(get_db),
):
    """Describe this image now and store the result, replacing any previous description.

    KEPT FOR OLD CLIENTS (a console tab loaded before console#564). It waits for the caption,
    as it always did, but the caption itself is a ticket now -- the same one
    POST /images/scene/describe would hand out -- so it is single-flight with every other
    caption of the image, and a dropped connection no longer loses the work.

    ALWAYS regenerates. This one call is both the first description and the re-roll, because
    they are the same act — the caller decides which it is by deciding whether to call.
    """
    _require_known_bucket(path)
    # WITHOUT A DATABASE CONNECTION while waiting (console#559): the auth lookup opened this
    # session's transaction, and the wait can be minutes behind other captions.
    await release_connection(db)
    asked = _request_halves(path, body)
    for t, _ in asked:
        await t.wait()
    by_half = {t.half: t for t, _ in asked}
    first = asked[0][0]
    scene = by_half.get(caption_tickets.SCENE)
    hard = scene if scene is not None else first
    if hard.status == caption_tickets.FAILED:
        # 404 for an image that cannot be read; 503 for the captioner, as /captions/describe
        # does -- it being down is a temporary condition on another host.
        raise HTTPException(status_code=404 if hard.unreadable else 503,
                            detail=hard.error or "the caption failed")
    motion = by_half.get(caption_tickets.MOTION)
    # A motion failure beside a saved scene is the old partial success, not an error.
    motion_error = (motion.error or "the motion caption failed") if (
        motion is not None and motion is not hard and motion.status == caption_tickets.FAILED
    ) else None
    meta = await db.get(ImageMeta, path, populate_existing=True)
    return _scene_response(path, meta, motion_error=motion_error)


@router.post("/images/scene/try", response_model=CaptionTryResponse,
             dependencies=[Depends(get_current_user)])
async def try_caption_prompts(
    path: str = Query(...),
    body: CaptionTryRequest | None = None,
    db: AsyncSession = Depends(get_db),
):
    """Run the caption and motion prompts from the Settings editors on one image (console#555).

    POST /images/scene WITHOUT THE WRITE. The Settings page calls it twice -- once with the
    unsaved editor text, once with nothing (the saved prompts) -- and shows the two side by
    side, so a prompt can be judged before it is saved. Storing either result would put a
    caption from a prompt nobody has adopted onto the image's record, where the next job
    would pick it up as the description.

    Not POST /captions/describe, although that is also a no-store preview: it produces the
    static half only, and the motion prompt -- grounded on the static half -- is half of
    what is being tried. This is the same two-call pair /images/scene makes, through the
    same function.

    THROUGH THE CAPTION QUEUE, same as /images/scene. A try is two describes on the one-slot
    captioner, and a person comparing prompts clicks "Try" repeatedly; outside the queue
    those calls would race the bulk-tag and dataset captioning that is already waiting, and
    the back of that line would start timing out (app/caption_queue.py). Same refusal rules
    too: a box that is rendering answers 503 with its name.
    """
    from app.caption_queue import queue as caption_queue
    from app.joycaption import instruction_for
    from app.routes.app_settings import _get_all_settings

    _require_known_bucket(path)
    body = body or CaptionTryRequest()

    # Resolved HERE rather than left to caption_image_pair, because "" means something
    # different in each place: to this request it is "the default", while caption_image_pair
    # would send an empty instruction to the captioner.
    cfg = await _get_all_settings(db)
    style = body.caption_style or cfg.get("caption_style", "")
    custom = (body.caption_instruction if body.caption_instruction is not None
              else cfg.get("caption_instruction", ""))
    instruction = instruction_for(style, custom)

    # Not holding a pooled connection while in line -- see describe_image_scene (console#559).
    await release_connection(db)
    async with caption_queue.turn(path, kind="try"):
        try:
            image = await asyncio.to_thread(download_bytes, path)
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"could not read {path}: {e}") from e
        try:
            # motion_template None falls through to the saved one and "" to the default
            # template -- caption_image_pair already reads it that way.
            pair = await caption_image_pair(
                db, image, instruction=instruction,
                motion_style=body.motion_style, motion_instruction=body.motion_template)
        except CaptionError as e:
            logger.warning("prompt try failed for %s: %s", path, e)
            raise HTTPException(status_code=503, detail=str(e)) from e

    if not pair.scene.strip():
        raise HTTPException(status_code=503,
                            detail="the captioner returned nothing for this image")
    logger.info("prompt try on %s: caption %d words, motion %s", path,
                len(pair.scene.split()),
                f"{len(pair.motion.split())} words" if pair.motion else
                (f"failed ({pair.motion_error})" if pair.motion_error else "off"))
    return CaptionTryResponse(
        caption=pair.scene,
        words=len(pair.scene.split()),
        motion=pair.motion,
        motion_words=len(pair.motion.split()) if pair.motion else 0,
        motion_error=pair.motion_error,
        motion_enabled=settings.motion_caption_enabled,
        caption_instruction_used=pair.scene_instruction,
        motion_instruction_used=pair.motion_instruction,
    )

def search_pattern(q: str) -> str:
    """The LIKE pattern for a user's query, with their wildcards neutralised.

    "%" and "_" are LIKE wildcards and filenames are full of both, so a query for "a_b" must not
    silently match "axb", and "100%" must not become match-everything.
    """
    return f"%{like_escape(q)}%"


def path_clause(q: str):
    """Match the S3 key as a substring.

    Fragment matching is right here and wrong for tags. Images arrive named
    "00111-1696092597-swapped.png" and get referred to by that number in job configs and notes,
    so "which folder was 00111 in" has to be answerable -- and a partially remembered filename is
    the only handle there is. Tags have exact controls of their own, so they no longer share this.
    """
    return ImageMeta.path.ilike(search_pattern(q), escape="\\")


def description_clause(q: str):
    """Match WHOLE WORDS inside a description, case-folded. None when the query is blank.

    Prose is words, and whole-word matching is the whole point here: `%red%` matched "textured"
    on production the day description search shipped (wanly-console#447's first pass) — red is a
    substring of texture, character, hundred and a dozen other words, so substring is the wrong
    shape for prose. A user typing "red" means the colour, and "textured skin" should not answer.

    ilike cannot express a word boundary, so this is a case-insensitive regular expression
    (`~*`) with boundaries around the escaped query. Escape for REGEX metacharacters, not LIKE
    ones: prose searches are words, and "(" or "?" in a description must not break the pattern.
    A known limit of \\y: it needs a word character on one side, so a query that begins or ends
    in punctuation ("(top-down)" or "100%") matches nothing through a description — the bare
    word ("top-down", "100") is the search that works, and that is the honest failure: a
    punctuation-edge query is a filename-shaped question, not a content one.

    Multi-word queries match the phrase as words joined by whitespace ("red dress"), which is
    the natural reading.
    """
    words = q.split()
    if not words:
        return None
    pattern = r"\y" + r"\s+".join(re.escape(w) for w in words) + r"\y"
    return ImageMeta.scene_description.op("~*")(pattern)


def tag_clause(tag: str):
    """Match one WHOLE tag inside an image's comma-joined tags string.

    Measured on production 2026-08-14, boundaries are the whole point: `%kelly%` matched 2,057 of
    2,788 images -- 74% of the repo -- because it also caught KellyYoung (1,019), KellyBangs (140)
    and KellyTeacher (76). Exact Kelly is 824. Jobs share the implementation via `app.tag_filter`.
    """
    return _tag_clause(ImageMeta.tags, tag)


def image_filter(q: str | None, tags: list[str], exclude: list[str]) -> list:
    """Every criterion ANDs. Returns clauses for .where(*clauses).

    Strict conjunction, deliberately: each pill narrows. Two subject pills therefore mean "both
    tags on one image", which is usually empty and is the honest answer -- there is no OR, so
    "the Kelly family" is two searches rather than one. That is the agreed v1 semantic.
    """
    clauses = [tag_clause(t) for t in tags if t.strip()]
    # NULL tags make NOT LIKE null, which would silently drop untagged images from every
    # excluded search. They have nothing to exclude, so they pass.
    clauses += [
        or_(ImageMeta.tags.is_(None), not_(tag_clause(t)))
        for t in exclude
        if t.strip()
    ]
    if q and q.strip():
        q = q.strip()
        # The description joins q under an OR with the filename (wanly-console#447): the
        # filename matches as a fragment, the description as whole words.
        clauses.append(or_(path_clause(q), description_clause(q)))
    return clauses


def repo_images_only():
    """Restrict a query over image_meta to the repo.

    image_meta is keyed by s3:// path and now also holds descriptions of GENERATED frames —
    a segment's last frame, described so a continuation can show the words before they are
    used (console#438). Those live in the jobs bucket and are not repo images. Without this,
    a filename search would HEAD one, find it, and put a render's intermediate frame in the
    Image Repo, which is why console#427 refused to store them at all.

    Training datasets share the images bucket and are excluded too: the folder listing hides
    the datasets/ prefix (wanly-console#464), so a search that still returned them would make
    the two look connected -- and with descriptions now searched by content (wanly-console#447),
    a described dataset staging copy would fill results from the repo search box. Narrows rather
    than excludes: repo folders keep their s3:// path, so one LIKE still covers every real image.

    Separate from image_filter deliberately: that function returns the USER's criteria, and
    an empty list is how the route knows nothing was asked for and answers 400 rather than
    serving the whole repo. Folding this in would make it never empty.
    """
    return and_(
        ImageMeta.path.like(f"s3://{settings.s3_images_bucket}/%"),
        not_(ImageMeta.path.like(f"s3://{settings.s3_images_bucket}/{DATASETS_PREFIX}/%")),
    )


@router.get("/images/search", dependencies=[Depends(get_current_user)])
async def search_images(
    q: str | None = Query(None, max_length=500),
    tags: list[str] = Query(default_factory=list),
    exclude: list[str] = Query(default_factory=list),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
    user = Depends(get_current_user),
):
    """Find images by whole tags (AND) and/or a filename or description fragment.

    Two controls with two different jobs. `tags` and `exclude` match a tag in full, so Kelly
    stops dragging in KellyYoung; `q` matches the S3 key or the description as a substring --
    a half-recalled filename is one handle on an image, what is IN the image is the other, and
    the filename is the only way an untagged, undescribed image can be found at all.

    Everything given ANDs. `tags=Kelly&tags=Missionary` is the 102 images carrying both, not the
    2,057 that a substring search for "kelly" used to return.
    """
    clauses = image_filter(q, tags, exclude)
    if not clauses:
        raise HTTPException(
            status_code=400,
            detail="Provide q, or at least one tag — an unfiltered search is the folder listing.",
        )

    count_q = select(func.count()).select_from(ImageMeta).where(repo_images_only(), *clauses)
    total = (await db.execute(count_q)).scalar() or 0

    meta_q = (
        select(ImageMeta)
        .where(repo_images_only(), *clauses)
        .order_by(ImageMeta.updated_at.desc())
        .offset(offset)
        .limit(limit)
    )
    meta_rows = (await db.execute(meta_q)).scalars().all()

    async def _meta(meta: ImageMeta) -> dict | None:
        obj = await asyncio.to_thread(head_object, meta.path)
        if not obj:
            return None
        key = obj["Key"]
        return {
            "key": key,
            "path": meta.path,
            "filename": key.split("/", 1)[1] if "/" in key else key,
            "size": obj["Size"],
            "last_modified": obj["LastModified"],
            **_meta_fields(meta),
        }

    # Concurrently, because this is the page's real cost. Each row needs one S3 HEAD, and a
    # sequential await per row made a 50-result page 50 serial round trips -- far more than the
    # query itself takes over 2,788 rows.
    results = await asyncio.gather(*(_meta(row) for row in meta_rows))
    items = [item for item in results if item]

    return {"items": items, "total": total, "limit": limit, "offset": offset}


@router.get("/images/tag-counts", dependencies=[Depends(get_current_user)])
async def image_tag_counts(
    q: str | None = Query(None, max_length=500),
    tags: list[str] = Query(default_factory=list),
    exclude: list[str] = Query(default_factory=list),
    db: AsyncSession = Depends(get_db),
    user = Depends(get_current_user),
):
    """Every tag in use, with how many images carry it under the CURRENT filter.

    Counts are what make the filter navigable rather than a guessing game: with Kelly selected,
    the remaining tags show what actually exists inside that set, so a dead end is visible before
    it is clicked instead of after.

    Derived from what is used, not from the title_tags vocabulary. Tagging is meant to be
    controlled, but production has drifted -- 11 tags in use are not in the vocabulary, including
    kellyteacher on 76 images. Driving the pills from the vocabulary would make those 76 images
    unreachable. It also surfaces the fat-fingers (`pusy`, `cowgirlowgirl`, one image each) with
    their counts, which is the first step to cleaning them up.
    """
    clauses = image_filter(q, tags, exclude)

    # unnest in a LATERAL rather than the target list: a set-returning function in the select
    # list cannot then be grouped by.
    tag_rows = (
        func.unnest(func.string_to_array(ImageMeta.tags, ","))
        .table_valued("tag")
        .render_derived(name="t")
    )
    tag_expr = func.lower(func.btrim(tag_rows.c.tag))

    stmt = (
        select(tag_expr.label("tag"), func.count().label("count"))
        .select_from(ImageMeta)
        .join(tag_rows, true())
        .where(repo_images_only(), *clauses, tag_expr != "")
        .group_by(tag_expr)
        .order_by(func.count().desc(), tag_expr)
    )
    rows = (await db.execute(stmt)).all()
    return {"items": [{"tag": r.tag, "count": r.count} for r in rows]}


_CONTENT_TYPES = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "webp": "image/webp"}


@router.get("/images/jobs", dependencies=[Depends(get_current_user)])
async def get_image_jobs(
    path: str = Query(...),
    db: AsyncSession = Depends(get_db),
    user = Depends(get_current_user),
):
    """Return jobs that use the given image as their starting image."""
    result = await db.execute(
        select(Job.id, Job.name, Job.created_at)
        .where(Job.user_id == user.id, Job.starting_image == path)
        .order_by(Job.created_at.desc())
        .limit(50)
    )
    rows = result.all()
    return [
        {"id": str(row[0]), "name": row[1], "created_at": row[2].isoformat()}
        for row in rows
    ]


@router.get("/images/download", dependencies=[Depends(verify_api_key_or_token)])
async def download_image_bytes(path: str = Query(...)):
    """Return raw image bytes for canvas processing in the browser.

    Unlike /files, this does not redirect to S3. Returning bytes directly means
    FastAPI's CORS middleware covers the response, so the console can fetch() the
    image and draw it to a canvas without triggering cross-origin taint.
    """
    bucket = settings.s3_images_bucket
    if not path.startswith(f"s3://{bucket}/"):
        raise HTTPException(status_code=400, detail="Path must be in the images bucket")
    try:
        data = await asyncio.to_thread(download_bytes, path)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Image not found: {e}")
    ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
    return Response(content=data, media_type=_CONTENT_TYPES.get(ext, "application/octet-stream"))
