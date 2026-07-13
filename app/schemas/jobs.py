from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field

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
    lightx2v_strength_high: Optional[float] = None
    lightx2v_strength_low: Optional[float] = None
    cfg_high: Optional[float] = None
    cfg_low: Optional[float] = None
    steps_total: Optional[int] = None
    high_noise_steps: Optional[int] = None
    flow_shift: Optional[float] = None
    video_preset_id: Optional[UUID] = None
    starting_image_uri: Optional[str] = None
    starting_image_hash: Optional[str] = None
    first_segment: SegmentCreate
    tags: Optional[str] = Field(None, max_length=500)


class JobLoraSummary(BaseModel):
    lora_id: Optional[str] = None
    name: Optional[str] = None
    high_file: Optional[str] = None
    low_file: Optional[str] = None
    high_weight: Optional[float] = None
    low_weight: Optional[float] = None


class JobResponse(BaseModel):
    id: UUID
    name: str
    width: int
    height: int
    fps: int
    seed: int
    starting_image: Optional[str]
    lightx2v_strength_high: Optional[float]
    lightx2v_strength_low: Optional[float]
    cfg_high: Optional[float]
    cfg_low: Optional[float]
    steps_total: Optional[int] = None
    high_noise_steps: Optional[int] = None
    flow_shift: Optional[float] = None
    priority: int
    config_starred: bool = False
    status: str
    segment_count: int = 0
    completed_segment_count: int = 0
    estimated_run_time: Optional[float] = None
    faceswap_enabled: bool = False
    loras: list[JobLoraSummary] = []
    tags: Optional[str] = None
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
    tags: Optional[str] = Field(None, max_length=500, description="Comma-separated tags")
    config_starred: Optional[bool] = Field(None, description="Flag this job's config as a successful one")
    video_preset_id: Optional[UUID] = Field(None, description="Job default video-settings preset")


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
