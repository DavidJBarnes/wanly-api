import asyncio
import logging
import re
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Form, HTTPException, Query, UploadFile
from fastapi.responses import Response
from sqlalchemy import func, not_, or_, select, true
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import get_current_user, verify_api_key_or_bearer, verify_api_key_or_token
from app.config import settings
from app.database import get_db
from app.joycaption import CaptionError
from app.models import Favorite, ImageMeta, Job, Segment, User
from app.routes.captions import caption_image_bytes
from app.schemas.images import ImageSceneRequest, ImageSceneResponse, ImageTagsUpdate
from app.tag_filter import like_escape
from app.tag_filter import tag_clause as _tag_clause
from app.s3 import (
    delete_object,
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
               ImageMeta.scene_description, ImageMeta.scene_described_at)
        .where(ImageMeta.path.in_(paths))
    )).all()
    return {
        row[0]: {
            "tags": row[1] or None,
            "scene_description": row[2] or None,
            "scene_described_at": row[3],
        }
        for row in rows
    }


_NO_META = {"tags": None, "scene_description": None, "scene_described_at": None}


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
        entry = refs.setdefault(path, {"job_ids": [], "segment_ids": []})
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

    return refs


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
        # Same helper the delete endpoint gates on. When these two disagree the UI is the one
        # that gets believed, which is how referenced images got deleted in the first place.
        in_use_set = set(await find_image_references(db, paths))
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


@router.get("/images/favorites", dependencies=[Depends(get_current_user)])
async def list_favorite_images(
    user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return all favorited images across all folders with metadata."""
    result = await db.execute(
        select(Favorite.item_ref)
        .where(Favorite.user_id == user.id, Favorite.item_type == "image")
        .order_by(Favorite.created_at.desc())
    )
    refs = [row[0] for row in result.all()]

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


@router.get("/images/untagged", dependencies=[Depends(get_current_user)])
async def list_untagged_images(
    db: AsyncSession = Depends(get_db),
):
    """Return all images across all folders that have no tags — for tagging triage.

    "Untagged" means no image_meta row at all, or a row with empty/whitespace tags.
    Since never-tagged images have no row, this is a cross-folder scan minus the
    set of paths with non-empty tags.
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
        if not obj["Key"].endswith("/.folder")
    ]
    paths = [f"s3://{bucket}/{obj['Key']}" for obj in objects]

    tagged: set[str] = set()
    in_use_set: set[str] = set()
    meta_map: dict[str, dict] = {}
    if paths:
        meta_map = await _meta_by_path(db, paths)
        tagged = {p for p, m in meta_map.items() if (m["tags"] or "").strip()}

        in_use_set = set(await find_image_references(db, paths))

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
        ))
        await db.delete(meta)
    await db.commit()

    return {"moved": len(moved)}


@router.delete("/images", dependencies=[Depends(get_current_user)])
async def delete_image(
    path: str = Query(...),
    force: bool = Query(False),
    db: AsyncSession = Depends(get_db),
):
    """Delete a single image by S3 URI, refusing while a job or segment still points at it.

    force=true skips the check for when the deletion is genuinely intended and the resulting
    dangling reference is accepted.
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
                },
            )
    await asyncio.to_thread(delete_object, path)
    return {"ok": True}


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


# ---------------------------------------------------------------------------------------
# Scene descriptions (console#414)
#
# JoyCaption's description of a frame, stored on the image rather than re-derived per job.
# POST /captions/describe stays as it is: a stateless preview for bytes nobody has committed
# to. This is the other case — an image that lives in the repo, described once, by a person
# who is looking at it.
# ---------------------------------------------------------------------------------------


def _scene_response(path: str, meta: ImageMeta | None) -> ImageSceneResponse:
    description = (meta.scene_description if meta else None) or None
    return ImageSceneResponse(
        path=path,
        scene_description=description,
        scene_instruction=(meta.scene_instruction if meta else None) or None,
        scene_described_at=meta.scene_described_at if meta else None,
        words=len(description.split()) if description else 0,
    )


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


@router.get("/images/scene", response_model=ImageSceneResponse,
            dependencies=[Depends(get_current_user)])
async def get_image_scene(path: str = Query(...), db: AsyncSession = Depends(get_db)):
    """This image's saved description, or nulls if it has never been described.

    Nulls rather than a 404: "no description yet" is an ordinary state of a perfectly good
    image, and the caller — the New Job modal — has to render it either way.
    """
    _require_known_bucket(path)
    return _scene_response(path, await db.get(ImageMeta, path))


@router.post("/images/scene", response_model=ImageSceneResponse,
             dependencies=[Depends(get_current_user)])
async def describe_image_scene(
    path: str = Query(...),
    body: ImageSceneRequest | None = None,
    db: AsyncSession = Depends(get_db),
):
    """Describe this image now and store the result, replacing any previous description.

    ALWAYS regenerates. This one call is both the first description and the re-roll, because
    they are the same act — the caller decides which it is by deciding whether to call. Any
    "only if missing" rule here would be a second opinion about a decision the UI has
    already made, and would make re-roll impossible to express.
    """
    _require_known_bucket(path)
    body = body or ImageSceneRequest()

    try:
        # boto3 is synchronous; off the event loop so one slow fetch cannot stall the API.
        image = await asyncio.to_thread(download_bytes, path)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"could not read {path}: {e}") from e

    try:
        caption, instruction = await caption_image_bytes(
            db, image, style=body.style, instruction=body.instruction)
    except CaptionError as e:
        # 503, as /captions/describe does: the captioner being down is a temporary condition
        # on another host, not a bug in this request. "Try again" is honest advice.
        logger.warning("scene description failed for %s: %s", path, e)
        raise HTTPException(status_code=503, detail=str(e)) from e

    caption = caption.strip()
    if not caption:
        # A blank caption is a failure wearing a success's clothes. Storing it would mark the
        # image described and stop anything ever asking again.
        raise HTTPException(status_code=503,
                            detail="the captioner returned nothing for this image")

    meta = await db.get(ImageMeta, path)
    if meta is None:
        meta = ImageMeta(path=path)
        db.add(meta)
    meta.scene_description = caption
    meta.scene_instruction = instruction
    meta.scene_described_at = datetime.now(timezone.utc)
    await db.commit()
    await db.refresh(meta)

    return _scene_response(path, meta)


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
        clauses.append(path_clause(q.strip()))
    return clauses


def repo_images_only():
    """Restrict a query over image_meta to the repo.

    image_meta is keyed by s3:// path and now also holds descriptions of GENERATED frames —
    a segment's last frame, described so a continuation can show the words before they are
    used (console#438). Those live in the jobs bucket and are not repo images. Without this,
    a filename search would HEAD one, find it, and put a render's intermediate frame in the
    Image Repo, which is why console#427 refused to store them at all.

    Separate from image_filter deliberately: that function returns the USER's criteria, and
    an empty list is how the route knows nothing was asked for and answers 400 rather than
    serving the whole repo. Folding this in would make it never empty.
    """
    return ImageMeta.path.like(f"s3://{settings.s3_images_bucket}/%")


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
    """Find images by whole tags (AND) and/or a filename fragment.

    Two controls with two different jobs. `tags` and `exclude` match a tag in full, so Kelly
    stops dragging in KellyYoung; `q` matches the S3 key as a substring, because a half-recalled
    filename is often the only handle there is, and it is the only way an untagged image can be
    found at all.

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
