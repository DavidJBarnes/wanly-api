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
    continuation_mode: Optional[str] = None  # "traditional" | "vace" (NULL -> global default)
    # Render with the character's identity reference -- its character sheet or face ref
    # (wanly-console#581)? None: yes when the character has one. False: not on this job.
    use_identity_ref: Optional[bool] = None
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


class CaptionHoldDetail(BaseModel):
    """What a caption-held job is waiting for, right now (console#562 follow-up).

    `needs`: the halves not saved yet, "scene" and/or "motion". `queue_status`: "queued"
    (with `queue_position`, 1 = next up), "running", "waiting" (its waiter is between
    refusals -- the box beside the captioner is rendering), or None (no waiter yet; the sweep
    gives it one within seconds). Computed per request, never stored.
    """
    image: Optional[str] = None
    needs: list[str] = []
    queue_status: Optional[str] = None
    queue_position: Optional[int] = None
    queue_depth: int = 0
    note: Optional[str] = None
    #: Which caption lane the place is in: "scene" or "motion" (wanly-console#572).
    lane: Optional[str] = None


class CaptionHoldImage(CaptionHoldDetail):
    """One held image in the summary: how many segments and jobs wait on it."""
    segments: int = 0
    jobs: int = 0


class CaptionHoldSummary(BaseModel):
    """GET /caption-holds: how many jobs are waiting on captions, and how the queue looks."""
    jobs_waiting: int = 0
    segments_waiting: int = 0
    jobs_failed: int = 0
    segments_failed: int = 0
    queue_depth: int = 0
    queue_waiting: int = 0
    running: Optional[str] = None
    #: The motion lane on its own (wanly-console#572); queue_* above count both lanes and
    #: `running` is the scene lane's.
    motion_queue_depth: int = 0
    motion_running: Optional[str] = None
    #: Held images, the next one to be captioned first.
    images: list[CaptionHoldImage] = []


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
    # The start frame's size, as the job was created. NOT necessarily what renders: see below.
    width: int
    height: int
    # The size the clips actually render at (#359). A recipe render is capped by the engine,
    # so an upscaled 1856x1280 start frame renders at 1216x832; everything else renders at
    # width x height and these equal them. Filled by the list, detail and reopen endpoints,
    # which are the ones that know whether a job's segments are recipe renders. None from the
    # endpoints that return the bare row (create, update, reorder): read it as width x height.
    render_width: Optional[int] = None
    render_height: Optional[int] = None
    fps: int
    seed: int
    starting_image: Optional[str]
    priority: int
    status: str
    segment_count: int = 0
    completed_segment_count: int = 0
    estimated_run_time: Optional[float] = None
    # "awaiting_caption" or "caption_failed" when a live segment is held on its start image's
    # caption (console#562), else None. The job's status stays "pending" -- it is queued --
    # so this is what says why it is not starting. Filled by the list and detail endpoints.
    caption_hold: Optional[str] = None
    # While caption_hold is "awaiting_caption": the image, what it still needs and its place
    # in the caption queue. Filled by the list and detail endpoints.
    caption_hold_detail: Optional[CaptionHoldDetail] = None
    tags: Optional[str] = None
    continuation_mode: Optional[str] = None
    use_identity_ref: Optional[bool] = None
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
    # The "Use character sheet" toggle (wanly-console#581). null restores the default.
    use_identity_ref: Optional[bool] = None


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
