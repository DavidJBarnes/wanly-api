import asyncio
import re
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Form, HTTPException, Query, UploadFile

from app.auth import get_current_user, verify_api_key_or_token
from app.config import settings
from app.s3 import (
    delete_object,
    get_first_object_key,
    list_common_prefixes,
    list_objects,
    upload_bytes,
)

router = APIRouter(tags=["images"])

_FOLDER_NAME_RE = re.compile(r"^[a-zA-Z0-9 _-]+$")


@router.post("/images/upload", dependencies=[Depends(verify_api_key_or_token)])
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


@router.delete("/images", dependencies=[Depends(get_current_user)])
async def delete_image(path: str = Query(...)):
    """Delete a single image by S3 URI."""
    bucket = settings.s3_images_bucket
    if not path.startswith(f"s3://{bucket}/"):
        raise HTTPException(status_code=400, detail="Path must be in the images bucket")
    await asyncio.to_thread(delete_object, path)
    return {"ok": True}
