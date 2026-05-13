import asyncio
import os
from uuid import UUID

from fastapi import UploadFile

from app.config import settings
from app.s3 import upload_bytes


async def upload_faceswap_image(
    file: UploadFile, job_id: UUID, key_suffix: str = "faceswap_source"
) -> str:
    """Upload a faceswap source image to S3 under the job prefix.

    Returns the S3 URI. Used by both create_job and reprocess_segment.
    """
    data = await file.read()
    ext = os.path.splitext(file.filename or "face.png")[1] or ".png"
    key = f"{job_id}/{key_suffix}{ext}"
    return await asyncio.to_thread(upload_bytes, data, key, settings.s3_jobs_bucket)
