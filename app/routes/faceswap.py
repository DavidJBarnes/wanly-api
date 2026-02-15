import asyncio
import logging
import os
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, status

from app.auth import get_current_user
from app.config import settings
from app.models import User
from app.s3 import _get_client

logger = logging.getLogger(__name__)

router = APIRouter()

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".avif", ".gif"}


def _list_face_objects() -> list[dict]:
    """List all image objects in the faces bucket, handling S3 pagination."""
    client = _get_client()
    bucket = settings.s3_faces_bucket
    objects: list[dict] = []
    kwargs: dict = {"Bucket": bucket}
    while True:
        resp = client.list_objects_v2(**kwargs)
        for obj in resp.get("Contents", []):
            key = obj["Key"]
            # Skip folder markers and non-image files
            if key.endswith("/") or obj.get("Size", 0) == 0:
                continue
            ext = os.path.splitext(key)[1].lower()
            if ext not in _IMAGE_EXTENSIONS:
                continue
            objects.append(obj)
        if not resp.get("IsTruncated"):
            break
        kwargs["ContinuationToken"] = resp["NextContinuationToken"]
    return objects


@router.get("/faceswap/presets")
async def list_faceswap_presets(
    _user: User = Depends(get_current_user),
):
    """List available faceswap preset face images from S3."""
    try:
        objects = await asyncio.to_thread(_list_face_objects)
    except Exception as exc:
        logger.exception("Failed to list faceswap presets from S3")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Unable to list faceswap presets from S3: {exc}",
        )

    bucket = settings.s3_faces_bucket
    presets = []
    for obj in objects:
        key = obj["Key"]
        s3_uri = f"s3://{bucket}/{key}"
        presets.append({
            "key": key,
            "name": os.path.splitext(os.path.basename(key))[0],
            "url": s3_uri,
            "thumbnail_url": f"/files?path={quote(s3_uri, safe='')}",
        })
    return presets
