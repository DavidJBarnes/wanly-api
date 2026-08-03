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
    # Re-anchor this segment's last frame to the faceswap face before it seeds the next
    # segment. Uses faceswap_image, so it only has an effect when faceswap is configured.
    seed_faceswap: bool = False
    negative_prompt: Optional[str] = None
    auto_finalize: bool = False
    transition: Optional[str] = None
    video_preset_id: Optional[UUID] = None


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
    seed_faceswap: bool = False
    negative_prompt: Optional[str] = None
    auto_finalize: bool
    transition: Optional[str]
    trim_start_frames: int
    trim_end_frames: int
    motion_keywords: Optional[list[str]] = None
    motion_magnitude: Optional[float] = None
    identity_mean_cos: Optional[float] = None
    identity_mean_cos_ref: Optional[float] = None
    identity_min_cos: Optional[float] = None
    identity_slope: Optional[float] = None
    identity_frames: Optional[int] = None
    identity_no_face: Optional[int] = None
    identity_face_px_p50: Optional[float] = None
    identity_yaw_max: Optional[float] = None
    identity_start_cos_ref: Optional[float] = None
    identity_end_cos_ref: Optional[float] = None
    identity_metrics: Optional[dict[str, Any]] = None
    reference_frames: Optional[list[str]] = None
    status: str
    reprocess_type: Optional[str] = None
    video_preset_id: Optional[UUID] = None
    worker_id: Optional[UUID]
    worker_name: Optional[str]
    output_path: Optional[str]
    last_frame_path: Optional[str]
    hologram_flavor: Optional[str] = None
    hologram_depth_scale_m: Optional[float] = None
    hologram_video_path: Optional[str] = None
    hologram_manifest_path: Optional[str] = None
    hologram_poster_path: Optional[str] = None
    # Lynx identity QA written by the daemon after a Lynx render (measurement only).
    lynx_identity_scores: Optional[dict[str, Any]] = None
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
    motion_magnitude: Optional[float] = None
    previous_motion_keywords: Optional[list[str]] = None
    previous_motion_magnitude: Optional[float] = None
    reference_frames: Optional[list[str]] = None
    lightx2v_strength_high: Optional[float] = None
    lightx2v_strength_low: Optional[float] = None
    cfg_high: Optional[float] = None
    cfg_low: Optional[float] = None
    steps_total: Optional[int] = None
    high_noise_steps: Optional[int] = None
    flow_shift: Optional[float] = None
    sampler_name: Optional[str] = None
    scheduler: Optional[str] = None
    negative_prompt: Optional[str] = None
    reprocess_type: Optional[str] = None
    output_path: Optional[str] = None
    width: int
    height: int
    fps: int
    seed: int
    # Continuation strategy for THIS segment, resolved API-side. "vace" only for
    # index>0 when enabled + the previous segment's video is available; else "traditional".
    continuation_mode: str = "traditional"
    previous_output_path: Optional[str] = None  # prev segment video (VACE control source)
    vace_overlap_frames: int = 12
    # Seed re-anchor: faceswap this segment's last frame to the canonical identity before it
    # seeds the next segment (resolved API-side: setting on AND a successor segment exists).
    seed_faceswap: bool = False
    # AR hologram (reprocess_type="ar_hologram"): the source is the job's finalized stitched
    # video (not this carrier segment's own output); params drive the daemon matte + manifest.
    hologram_source_path: Optional[str] = None
    hologram_key_color: Optional[str] = None
    hologram_subject_height_m: Optional[float] = None
    hologram_flavor: Optional[str] = None
    hologram_depth_scale_m: Optional[float] = None
    # Foundry smashcut (reprocess_type="smashcut_concat"): ordered source clip paths + transition.
    smashcut_clip_paths: Optional[list[str]] = None
    smashcut_transition: Optional[str] = None
    smashcut_clip_speeds: Optional[list[float]] = None
    # === Lynx identity-preserving engine (resolved from the job) ===
    # generation_engine="lynx" routes the daemon to build_lynx_workflow. Each lynx_*
    # value is a per-job override; None means "use the daemon's settings default".
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

    model_config = {"from_attributes": True}


class SmashcutRequest(BaseModel):
    """Build a smashcut: hard-cut concat of hand-picked segment clips (same resolution)."""
    name: str
    segment_ids: list[UUID]  # ordered
    transition: str = "seamless"  # "seamless" (butt-splice) | "black" (dip-to-black)
    # Per-clip playback speed, aligned 1:1 with segment_ids. Omit (or send all 1.0) for no
    # retiming. <1 is slow-motion, >1 is fast-forward.
    clip_speeds: Optional[list[float]] = None


class SegmentClipResponse(BaseModel):
    """A segment surfaced as a pickable clip in the Foundry pool / smashcut builder."""
    id: UUID
    job_id: UUID
    job_name: str
    index: int
    output_path: Optional[str] = None
    thumbnail_path: Optional[str] = None  # last_frame_path
    width: int
    height: int
    fps: int
    duration_seconds: float
    motion_magnitude: Optional[float] = None
    favorite: bool = False

    model_config = {"from_attributes": True}


class SegmentTrimUpdate(BaseModel):
    trim_start_frames: int = Field(ge=0)
    trim_end_frames: int = Field(ge=0)


class SegmentVideoPresetUpdate(BaseModel):
    video_preset_id: Optional[UUID] = None


class FramePreview(BaseModel):
    frame_index: int
    data_url: str


class FramePreviewResponse(BaseModel):
    total_frames: int
    fps: float
    frames: list[FramePreview]


class SegmentReprocessRequest(BaseModel):
    faceswap_enabled: bool = True
    faceswap_method: Optional[str] = None
    faceswap_source_type: Optional[str] = None
    faceswap_image: Optional[str] = None
    faceswap_faces_order: Optional[str] = None
    faceswap_faces_index: Optional[str] = None


class HologramRequest(BaseModel):
    """Per-request overrides for a 'Make Hologram' action. Unset -> AppSetting -> hardcoded default."""

    subject_height_m: Optional[float] = None
    key_color: Optional[str] = None
    flavor: Optional[str] = None  # "2d_matte" (default) or "2.5d_depth"
    depth_scale_m: Optional[float] = None  # 2.5d relief depth in meters (clamped 0.03..0.60)


class SegmentStatusUpdate(BaseModel):
    status: Optional[str] = None
    output_path: Optional[str] = None
    last_frame_path: Optional[str] = None
    error_message: Optional[str] = None
    progress_log: Optional[str] = None
    motion_keywords: Optional[list[str]] = None
    motion_magnitude: Optional[float] = None
    identity_mean_cos: Optional[float] = None
    identity_mean_cos_ref: Optional[float] = None
    identity_min_cos: Optional[float] = None
    identity_slope: Optional[float] = None
    identity_frames: Optional[int] = None
    identity_no_face: Optional[int] = None
    identity_face_px_p50: Optional[float] = None
    identity_yaw_max: Optional[float] = None
    identity_start_cos_ref: Optional[float] = None
    identity_end_cos_ref: Optional[float] = None
    identity_metrics: Optional[dict[str, Any]] = None
    vace_overlap_seconds: Optional[float] = None
    # Lynx identity QA measured by the daemon. Measurement only — no gating.
    lynx_identity_scores: Optional[dict[str, Any]] = None
