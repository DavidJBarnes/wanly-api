from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel

from app.schemas.segments import SegmentCreate, SegmentResponse
from app.schemas.videos import VideoResponse


class JobCreate(BaseModel):
    name: str
    width: int
    height: int
    fps: int
    seed: Optional[int] = None
    starting_image: Optional[str] = None
    first_segment: SegmentCreate


class JobResponse(BaseModel):
    id: UUID
    name: str
    width: int
    height: int
    fps: int
    seed: int
    starting_image: Optional[str]
    status: str
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class JobDetailResponse(JobResponse):
    segments: list[SegmentResponse]
    videos: list[VideoResponse]
    segment_count: int
    completed_segment_count: int
    total_run_time: float
    total_video_time: float


class JobUpdate(BaseModel):
    name: Optional[str] = None
    status: Optional[str] = None
