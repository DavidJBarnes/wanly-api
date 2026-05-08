from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import get_current_user
from app.database import get_db
from app.models import Job, User, Video
from app.schemas.videos import VideoResponse

router = APIRouter()


class VideoTagsUpdate(BaseModel):
    tags: str | None = None


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
    )
    video = result.scalar_one_or_none()
    if video is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video not found")

    video.tags = body.tags
    await db.commit()
    await db.refresh(video)
    return video
