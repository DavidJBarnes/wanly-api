import asyncio
import re
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Form, HTTPException, Query, UploadFile
from fastapi.responses import Response
from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import get_current_user, verify_api_key_or_bearer, verify_api_key_or_token
from app.config import settings
from app.database import get_db
from app.models import Favorite, ImageMeta, Job, Segment
from app.schemas.images import ImageTagsUpdate
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

router = APIRouter(tags=["images"])

_FOLDER_NAME_RE = re.compile(r"^[a-zA-Z0-9 _-]+$")

# Every column that can hold an s3:// path a user could delete through this router.
# Job.identity_reference_image is deliberately absent: the daemon writes it into the *jobs*
# bucket, so it can never be the target of DELETE /images.
_JOB_IMAGE_COLUMNS = (Job.starting_image, Job.lynx_subject_image)
_SEGMENT_IMAGE_COLUMNS = (Segment.start_image, Segment.faceswap_image)


async def find_image_references(db: AsyncSession, paths: list[str]) -> dict[str, dict[str, list[str]]]:
    """Map each still-referenced path to the jobs and segments holding it.

    Deleting a referenced image fails silently: nothing breaks until a worker claims the
    segment weeks later and S3 returns 404, which costs a pickup and shows up as a red
    segment nowhere near the cause. That makes the check wider than it first looks:

      - segment-level refs count, not just Job.starting_image. The validated identity recipe
        sets Segment.faceswap_image on every job, so that is the common holder, and it was
        invisible to the old listing query.
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
    tags_map: dict[str, str] = {}
    if paths:
        # Same helper the delete endpoint gates on. When these two disagree the UI is the one
        # that gets believed, which is how referenced images got deleted in the first place.
        in_use_set = set(await find_image_references(db, paths))

        meta_result = await db.execute(
            select(ImageMeta.path, ImageMeta.tags).where(ImageMeta.path.in_(paths))
        )
        for row in meta_result.all():
            if row[1]:
                tags_map[row[0]] = row[1]

    return [
        {
            "key": obj["Key"],
            "path": f"s3://{bucket}/{obj['Key']}",
            "filename": obj["Key"].split("/", 1)[1] if "/" in obj["Key"] else obj["Key"],
            "size": obj["Size"],
            "last_modified": obj["LastModified"],
            "in_use": f"s3://{bucket}/{obj['Key']}" in in_use_set,
            "tags": tags_map.get(f"s3://{bucket}/{obj['Key']}"),
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
    if uris:
        meta_result = await db.execute(
            select(ImageMeta.path, ImageMeta.tags).where(ImageMeta.path.in_(uris))
        )
        tags_map = {row[0]: row[1] for row in meta_result.all() if row[1]}
        for item in items:
            if item is not None:
                item["tags"] = tags_map.get(item["path"])

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
    if paths:
        meta_result = await db.execute(
            select(ImageMeta.path, ImageMeta.tags).where(ImageMeta.path.in_(paths))
        )
        tagged = {row[0] for row in meta_result.all() if row[1] and row[1].strip()}

        in_use_set = set(await find_image_references(db, paths))

    untagged = [
        {
            "key": obj["Key"],
            "path": f"s3://{bucket}/{obj['Key']}",
            "filename": obj["Key"].split("/", 1)[1] if "/" in obj["Key"] else obj["Key"],
            "size": obj["Size"],
            "last_modified": obj["LastModified"],
            "in_use": f"s3://{bucket}/{obj['Key']}" in in_use_set,
            "tags": None,
        }
        for obj in objects
        if f"s3://{bucket}/{obj['Key']}" not in tagged
    ]
    untagged.sort(key=lambda x: x["last_modified"], reverse=True)
    return untagged


@router.post("/images/move", dependencies=[Depends(get_current_user)])
async def move_images(body: dict):
    """Move one or more images to a target folder (S3 copy + delete)."""
    keys: list[str] = body.get("keys", [])
    target_folder: str = body.get("target_folder", "").strip()
    if not keys:
        raise HTTPException(status_code=400, detail="No keys provided")
    if not target_folder:
        raise HTTPException(status_code=400, detail="target_folder is required")
    bucket = settings.s3_images_bucket

    async def _move_one(src_key: str) -> str:
        filename = src_key.split("/", 1)[1] if "/" in src_key else src_key
        dst_key = f"{target_folder}/{filename}"
        await asyncio.to_thread(move_object, bucket, src_key, dst_key)
        return dst_key

    moved = await asyncio.gather(*[_move_one(k) for k in keys])
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

    if tags_val is None:
        if meta:
            await db.delete(meta)
    else:
        if meta:
            meta.tags = tags_val
        else:
            meta = ImageMeta(path=path, tags=tags_val)
            db.add(meta)

    await db.commit()
    return {"path": path, "tags": tags_val}


def search_pattern(q: str) -> str:
    """The LIKE pattern for a user's query, with their wildcards neutralised.

    "%" and "_" are LIKE wildcards and filenames are full of both, so a query for "a_b" must not
    silently match "axb", and "100%" must not become match-everything. The backslash is escaped
    first, or escaping the wildcards would itself be re-escaped.
    """
    safe = q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{safe}%"


def search_clause(q: str):
    """Match a query against tags OR filename.

    A disjunction, not a conjunction: an untagged image has NULL tags, and requiring both would
    leave it permanently unfindable -- which is the case this exists for.
    """
    pattern = search_pattern(q)
    return or_(
        ImageMeta.tags.ilike(pattern, escape="\\"),
        ImageMeta.path.ilike(pattern, escape="\\"),
    )


@router.get("/images/search", dependencies=[Depends(get_current_user)])
async def search_images(
    q: str = Query(..., min_length=1, max_length=500),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
    user = Depends(get_current_user),
):
    """Search images across all folders by tag or filename (case-insensitive partial match).

    Path is matched as well as tags because the filename is often the only handle a user has.
    Images arrive named like "00111-1696092597-swapped.png", get referenced by that name in job
    configs, notes and conversations, and are spread across dated folders -- so "which folder was
    00111 in" was unanswerable from the UI, and an untagged image was unfindable by any means at
    all. Matching the path also makes a folder name a usable query, which is a free side effect
    of storing the full key.
    """
    matches = search_clause(q)

    count_q = select(func.count()).select_from(ImageMeta).where(matches)
    total = (await db.execute(count_q)).scalar() or 0

    meta_q = (
        select(ImageMeta)
        .where(matches)
        .order_by(ImageMeta.updated_at.desc())
        .offset(offset)
        .limit(limit)
    )
    meta_rows = (await db.execute(meta_q)).scalars().all()

    async def _meta(meta: ImageMeta) -> dict:
        bucket = settings.s3_images_bucket
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
            "tags": meta.tags,
        }

    items = []
    for row in meta_rows:
        item = await _meta(row)
        if item:
            items.append(item)

    return {"items": items, "total": total, "limit": limit, "offset": offset}


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
