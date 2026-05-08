from uuid import UUID

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import get_current_user
from app.database import get_db
from app.models import Job, User, Video
from app.schemas.videos import VideoResponse

logger = logging.getLogger(__name__)

router = APIRouter()


class VideoTagsUpdate(BaseModel):
    tags: str | None = Field(None, max_length=500, description="Comma-separated tags")


@router.patch("/videos/{video_id}/tags", response_model=VideoResponse)
async def update_video_tags(
    video_id: UUID,
    body: VideoTagsUpdate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Video)
        .join(Job, Video.job_id == Job.id)
        .where(Video.id == video_id, Job.user_id == user.id)
        .with_for_update()
    )
    video = result.scalar_one_or_none()
    if video is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video not found")

    try:
        video.tags = body.tags
        await db.commit()
        await db.refresh(video)
    except Exception as e:
        await db.rollback()
        logger.error("Failed to update tags for video %s: %s", video_id, e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to update tags")

    return video
