import asyncio
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, UploadFile

from app.auth import verify_api_key
from app.config import settings
from app.s3 import upload_bytes

router = APIRouter(tags=["images"], dependencies=[Depends(verify_api_key)])


@router.post("/images/upload")
async def upload_image(file: UploadFile, filename: str | None = None):
    data = await file.read()
    if not filename:
        ext = ""
        if file.filename and "." in file.filename:
            ext = "." + file.filename.rsplit(".", 1)[1]
        else:
            ext = ".png"
        filename = f"{uuid.uuid4().hex}{ext}"
    date_prefix = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    key = f"{date_prefix}/{filename}"
    bucket = settings.s3_images_bucket
    uri = await asyncio.to_thread(upload_bytes, data, key, bucket)
    return {"path": uri}
