from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel

from app.schemas.segments import SegmentCreate, SegmentResponse
from app.schemas.videos import VideoResponse


class JobReorderRequest(BaseModel):
    job_ids: list[UUID]


class JobCreate(BaseModel):
    name: str
    width: int
    height: int
    fps: int
    seed: Optional[int] = None
    first_segment: SegmentCreate


class JobResponse(BaseModel):
    id: UUID
    name: str
    width: int
    height: int
    fps: int
    seed: int
    starting_image: Optional[str]
    priority: int
    status: str
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class JobListResponse(BaseModel):
    items: list[JobResponse]
    total: int
    limit: int
    offset: int


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


class WorkerStatsItem(BaseModel):
    worker_name: str
    segments_completed: int
    avg_run_time: float
    last_seen: Optional[datetime] = None


class StatsResponse(BaseModel):
    jobs_by_status: dict[str, int]
    segments_by_status: dict[str, int]
    avg_segment_run_time: Optional[float]
    total_segments_completed: int
    total_video_time: float
    worker_stats: list[WorkerStatsItem]
