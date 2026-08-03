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
    continuation_mode: Optional[str] = None  # "traditional" | "vace" (NULL -> global default)
    # === Lynx identity-preserving engine ===
    # generation_engine="lynx" routes the job to the Lynx graph builder. Every lynx_*
    # tunable is optional: None -> the daemon's settings default (the same
    # per-job-override precedence the sampler params above use).
    generation_engine: Optional[str] = None
    lynx_subject_image: Optional[str] = None
    lynx_ip_scale: Optional[float] = None
    lynx_ref_scale: Optional[float] = None
    lynx_cfg_scale: Optional[float] = None
    lynx_start_percent: Optional[float] = None
    lynx_end_percent: Optional[float] = None
    lynx_ref_blocks_to_use: Optional[str] = None
    lynx_ip_layers: Optional[str] = None
    lynx_resampler: Optional[str] = None
    lynx_steps: Optional[int] = None
    lynx_cfg: Optional[float] = None
    lynx_shift: Optional[float] = None
    lynx_scheduler: Optional[str] = None
    lynx_distill_strength: Optional[float] = None
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
    video_preset_id: Optional[UUID] = None
    continuation_mode: Optional[str] = None
    # === Lynx identity-preserving engine ===
    # generation_engine="lynx" routes the job to the Lynx graph builder. Every lynx_*
    # tunable is optional: None -> the daemon's settings default (the same
    # per-job-override precedence the sampler params above use).
    generation_engine: Optional[str] = None
    lynx_subject_image: Optional[str] = None
    lynx_ip_scale: Optional[float] = None
    lynx_ref_scale: Optional[float] = None
    lynx_cfg_scale: Optional[float] = None
    lynx_start_percent: Optional[float] = None
    lynx_end_percent: Optional[float] = None
    lynx_ref_blocks_to_use: Optional[str] = None
    lynx_ip_layers: Optional[str] = None
    lynx_resampler: Optional[str] = None
    lynx_steps: Optional[int] = None
    lynx_cfg: Optional[float] = None
    lynx_shift: Optional[float] = None
    lynx_scheduler: Optional[str] = None
    lynx_distill_strength: Optional[float] = None
    identity_reference_image: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class JobListResponse(BaseModel):
    items: list[JobResponse]
    total: int
    limit: int
    offset: int


class IdentityAggregate(BaseModel):
    """Job-level identity, derived from the per-segment scores rather than re-measuring.

    Frame-weighted so a 20-frame segment does not count the same as a 200-frame one, and it
    names the worst-drifting segment because "which segment went wrong" is the actionable
    answer - a blended average hides it.
    """
    mean_cos: Optional[float] = None
    mean_cos_ref: Optional[float] = None
    slope: Optional[float] = None
    frames: int = 0
    no_face: int = 0
    scored_segments: int = 0
    worst_segment_index: Optional[int] = None
    worst_segment_slope: Optional[float] = None
    # Lowest per-segment cumulative score in the job, and where it happened. A job can
    # average well while a late segment has lost the character entirely - each segment is
    # measured against its OWN start frame, which for a continuation is the previous
    # segment's already-drifted last frame. Observed: seg0 0.764, seg1 0.785 vs their own
    # starts (both "healthy") while seg1 sat at 0.544 vs the job's original image.
    min_cos_ref: Optional[float] = None
    min_cos_ref_segment_index: Optional[int] = None


class JobDetailResponse(JobResponse):
    segments: list[SegmentResponse]
    videos: list[VideoResponse]
    segment_count: int
    completed_segment_count: int
    total_run_time: float
    total_video_time: float
    identity: Optional[IdentityAggregate] = None


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
    # Windowed rather than lifetime: a rolling average over every segment ever run stops
    # moving, so it says nothing about how the rig is performing now.
    avg_segment_run_time_24h: Optional[float]
    # Estimated seconds of work still queued: every active segment of every active job,
    # priced with the same estimator the job queue uses.
    total_queue_time: float
    worker_stats: list[WorkerStatsItem]
