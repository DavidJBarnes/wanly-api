import asyncio
import logging
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.models import Job, Segment, Video
from app.s3 import download_bytes, upload_bytes
from app.schemas.segments import SegmentResponse
from app.stitch import stitch_video

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    job_id: UUID | None = None,
    filename: str | None = None,
):
    """Upload a file to S3. Optionally scoped to a job_id folder.

    Used by the console to upload starting images, faceswap source images, etc.
    Returns the S3 URI of the uploaded file.
    """
    data = await file.read()
    name = filename or file.filename or "upload"

    if job_id:
        key = f"{job_id}/{name}"
    else:
        key = f"uploads/{name}"

    uri = await asyncio.to_thread(upload_bytes, data, key, settings.s3_jobs_bucket)
    return {"path": uri}


@router.get("/files")
async def download_file(path: str):
    """Download a file from S3 by its S3 URI.

    Used by daemons to fetch start images and faceswap images without
    needing AWS credentials.
    """
    if not path.startswith("s3://"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Path must be an S3 URI (s3://...)",
        )

    try:
        data = await asyncio.to_thread(download_bytes, path)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File not found: {e}",
        )

    # Infer content type from extension
    ext_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".avif": "image/avif",
        ".gif": "image/gif",
        ".mp4": "video/mp4",
        ".safetensors": "application/octet-stream",
    }
    ext = "." + path.rsplit(".", 1)[-1].lower() if "." in path else ""
    media_type = ext_map.get(ext, "application/octet-stream")

    return Response(content=data, media_type=media_type)


@router.post("/segments/{segment_id}/upload", response_model=SegmentResponse)
async def upload_segment_output(
    segment_id: UUID,
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    last_frame: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """Upload segment output files (video + last frame).

    Called by daemons after processing a segment. Uploads to S3 and updates
    the segment record with paths and completed status.
    """
    segment = await db.get(Segment, segment_id)
    if segment is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Segment not found")

    video_data = await video.read()
    frame_data = await last_frame.read()

    video_key = f"{segment.job_id}/{segment.index}_output.mp4"
    frame_key = f"{segment.job_id}/{segment.index}_last_frame.png"

    video_uri, frame_uri = await asyncio.gather(
        asyncio.to_thread(upload_bytes, video_data, video_key, settings.s3_jobs_bucket),
        asyncio.to_thread(upload_bytes, frame_data, frame_key, settings.s3_jobs_bucket),
    )

    segment.output_path = video_uri
    segment.last_frame_path = frame_uri
    segment.status = "completed"

    from datetime import datetime, timezone
    segment.completed_at = datetime.now(timezone.utc)

    # Check if job needs status update (same logic as PATCH /segments/{id})
    job = await db.get(Job, segment.job_id)
    result = await db.execute(
        select(Segment).where(
            Segment.job_id == job.id,
            Segment.status.in_(["pending", "claimed", "processing"]),
        )
    )
    active_segments = result.scalars().all()
    if len(active_segments) == 0:
        if segment.auto_finalize:
            job.status = "finalized"
            video_record = Video(job_id=job.id, status="pending")
            db.add(video_record)
            await db.flush()
            background_tasks.add_task(stitch_video, video_record.id, job.id)
        else:
            job.status = "awaiting"

    await db.commit()
    await db.refresh(segment)
    return segment
