import asyncio
import logging
import os
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.auth import get_current_user
from app.config import settings
from app.models import User
from app.s3 import _get_client

logger = logging.getLogger(__name__)

router = APIRouter()


def _list_face_objects() -> list[dict]:
    """List all objects in the faces bucket and return preset metadata."""
    client = _get_client()
    bucket = settings.s3_faces_bucket
    resp = client.list_objects_v2(Bucket=bucket)
    return resp.get("Contents", [])


@router.get("/faceswap/presets")
async def list_faceswap_presets(
    request: Request,
    _user: User = Depends(get_current_user),
):
    """List available faceswap preset face images from S3."""
    try:
        objects = await asyncio.to_thread(_list_face_objects)
    except Exception:
        logger.exception("Failed to list faceswap presets from S3")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Unable to list faceswap presets from S3",
        )

    bucket = settings.s3_faces_bucket
    base_url = str(request.base_url).rstrip("/")
    presets = []
    for obj in objects:
        key = obj["Key"]
        s3_uri = f"s3://{bucket}/{key}"
        presets.append({
            "key": key,
            "name": os.path.splitext(os.path.basename(key))[0],
            "url": s3_uri,
            "thumbnail_url": f"{base_url}/files?path={quote(s3_uri, safe='')}",
        })
    return presets
