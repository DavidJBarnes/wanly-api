import asyncio
import re
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Form, HTTPException, Query, UploadFile
from fastapi.responses import Response

from app.auth import get_current_user, verify_api_key_or_bearer, verify_api_key_or_token
from app.config import settings
from app.s3 import (
    delete_object,
    download_bytes,
    get_first_object_key,
    list_common_prefixes,
    list_objects,
    move_object,
    upload_bytes,
)

router = APIRouter(tags=["images"])

_FOLDER_NAME_RE = re.compile(r"^[a-zA-Z0-9 _-]+$")


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


@router.get("/images/folders", dependencies=[Depends(get_current_user)])
async def list_folders():
    """List date folders in the images bucket, newest first."""
    bucket = settings.s3_images_bucket
    prefixes = await asyncio.to_thread(list_common_prefixes, bucket)
    # Sort newest first (prefixes look like "2026-02-27/")
    prefixes.sort(reverse=True)

    # Get first object in each folder for thumbnails (parallel)
    async def _thumb(prefix: str) -> dict:
        name = prefix.rstrip("/")
        key = await asyncio.to_thread(get_first_object_key, bucket, prefix)
        thumbnail = f"s3://{bucket}/{key}" if key else None
        return {"name": name, "thumbnail": thumbnail}

    folders = await asyncio.gather(*[_thumb(p) for p in prefixes])
    return list(folders)


@router.get("/images/folder/{date}", dependencies=[Depends(get_current_user)])
async def list_folder_images(date: str):
    """List images in a date folder."""
    bucket = settings.s3_images_bucket
    prefix = f"{date}/"
    objects = await asyncio.to_thread(list_objects, bucket, prefix)
    return [
        {
            "key": obj["Key"],
            "path": f"s3://{bucket}/{obj['Key']}",
            "filename": obj["Key"].split("/", 1)[1] if "/" in obj["Key"] else obj["Key"],
            "size": obj["Size"],
            "last_modified": obj["LastModified"],
        }
        for obj in objects
        if not obj["Key"].endswith("/.folder")
    ]


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
async def delete_image(path: str = Query(...)):
    """Delete a single image by S3 URI."""
    bucket = settings.s3_images_bucket
    if not path.startswith(f"s3://{bucket}/"):
        raise HTTPException(status_code=400, detail="Path must be in the images bucket")
    await asyncio.to_thread(delete_object, path)
    return {"ok": True}


_CONTENT_TYPES = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "webp": "image/webp"}


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
