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
    mode: str = "identity"  # GenerationMode — locked for all segments; daemon resolves the preset
    starting_image_uri: Optional[str] = None
    starting_image_hash: Optional[str] = None
    first_segment: SegmentCreate
    tags: Optional[str] = Field(None, max_length=500)


class JobResponse(BaseModel):
    id: UUID
    name: str
    width: int
    height: int
    fps: int
    seed: int
    starting_image: Optional[str]
    mode: str = "identity"
    kind: str = "generate"
    source_job_id: Optional[UUID] = None
    priority: int
    status: str
    segment_count: int = 0
    completed_segment_count: int = 0
    estimated_run_time: Optional[float] = None
    faceswap_enabled: bool = False
    tags: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class JobListResponse(BaseModel):
    items: list[JobResponse]
    total: int
    limit: int
    offset: int


class FinalCutCreate(BaseModel):
    # Final Cut now swaps a character's face onto the finalized video via FaceFusion.
    reference_image_uri: Optional[str] = None  # face to swap in; default = source job's starting_image
    face_index: int = 0          # which face to swap in a multi-person clip (0=first, left-to-right)
    distance: Optional[float] = None  # FaceFusion reference-face-distance; None -> daemon auto


class FinalCutSummary(BaseModel):
    id: UUID
    name: str
    kind: str = "final_cut"
    status: str
    created_at: datetime
    video: Optional[VideoResponse] = None

    model_config = {"from_attributes": True}


class JobDetailResponse(JobResponse):
    segments: list[SegmentResponse]
    videos: list[VideoResponse]
    final_cuts: list[FinalCutSummary] = []
    segment_count: int
    completed_segment_count: int
    total_run_time: float
    total_video_time: float


class JobUpdate(BaseModel):
    name: Optional[str] = None
    status: Optional[str] = None
    tags: Optional[str] = Field(None, max_length=500, description="Comma-separated tags")


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
