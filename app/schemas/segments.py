from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field, model_validator


class SegmentCreate(BaseModel):
    prompt: str
    duration_seconds: float = 5.0
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    start_image: Optional[str] = None
    loras: Optional[list[Any]] = None
    faceswap_enabled: bool = False
    faceswap_method: Optional[str] = None
    faceswap_source_type: Optional[str] = None
    faceswap_image: Optional[str] = None
    faceswap_faces_order: Optional[str] = None
    faceswap_faces_index: Optional[str] = None
    auto_finalize: bool = False
    transition: Optional[str] = None


class SegmentResponse(BaseModel):
    id: UUID
    job_id: UUID
    index: int
    prompt: str
    prompt_template: Optional[str]
    duration_seconds: float
    speed: float
    start_image: Optional[str]
    loras: Optional[list[Any]]
    faceswap_enabled: bool
    faceswap_method: Optional[str]
    faceswap_source_type: Optional[str]
    faceswap_image: Optional[str]
    faceswap_faces_order: Optional[str]
    faceswap_faces_index: Optional[str]
    auto_finalize: bool
    transition: Optional[str]
    trim_start_frames: int
    trim_end_frames: int
    motion_keywords: Optional[list[str]] = None
    reference_frames: Optional[list[str]] = None
    status: str
    worker_id: Optional[UUID]
    worker_name: Optional[str]
    output_path: Optional[str]
    last_frame_path: Optional[str]
    created_at: datetime
    claimed_at: Optional[datetime]
    completed_at: Optional[datetime]
    error_message: Optional[str]
    progress_log: Optional[str]
    estimated_run_time: Optional[float] = None

    model_config = {"from_attributes": True}


class WorkerSegmentResponse(BaseModel):
    id: UUID
    job_id: UUID
    job_name: str
    index: int
    prompt: str
    status: str
    duration_seconds: float
    created_at: datetime
    claimed_at: Optional[datetime]
    completed_at: Optional[datetime]


class SegmentClaimResponse(BaseModel):
    id: UUID
    job_id: UUID
    index: int
    prompt: str
    duration_seconds: float
    speed: float
    start_image: Optional[str]
    loras: Optional[list[Any]]
    faceswap_enabled: bool
    faceswap_method: Optional[str]
    faceswap_source_type: Optional[str]
    faceswap_image: Optional[str]
    faceswap_faces_order: Optional[str]
    faceswap_faces_index: Optional[str]
    initial_reference_image: Optional[str] = None
    motion_keywords: Optional[list[str]] = None
    previous_motion_keywords: Optional[list[str]] = None
    reference_frames: Optional[list[str]] = None
    lightx2v_strength_high: Optional[float] = None
    lightx2v_strength_low: Optional[float] = None
    cfg_high: Optional[float] = None
    cfg_low: Optional[float] = None
    negative_prompt: Optional[str] = None
    width: int
    height: int
    fps: int
    seed: int

    model_config = {"from_attributes": True}


class SegmentTrimUpdate(BaseModel):
    trim_start_frames: int = Field(ge=0)
    trim_end_frames: int = Field(ge=0)


class FramePreview(BaseModel):
    frame_index: int
    data_url: str


class FramePreviewResponse(BaseModel):
    total_frames: int
    fps: float
    frames: list[FramePreview]


class SegmentStatusUpdate(BaseModel):
    status: Optional[str] = None
    output_path: Optional[str] = None
    last_frame_path: Optional[str] = None
    error_message: Optional[str] = None
    progress_log: Optional[str] = None
    motion_keywords: Optional[list[str]] = None
