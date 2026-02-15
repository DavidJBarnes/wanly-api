import asyncio
import logging
import os

from fastapi import APIRouter, Depends

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
    objects = resp.get("Contents", [])
    presets = []
    for obj in objects:
        key = obj["Key"]
        name = os.path.splitext(os.path.basename(key))[0]
        presets.append({
            "key": key,
            "name": name,
            "url": f"s3://{bucket}/{key}",
        })
    return presets


@router.get("/faceswap/presets")
async def list_faceswap_presets(
    _user: User = Depends(get_current_user),
):
    """List available faceswap preset face images from S3."""
    presets = await asyncio.to_thread(_list_face_objects)
    return presets
