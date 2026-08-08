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
    faceswap_model: Optional[str] = None
    faceswap_pixel_boost: Optional[str] = None
    # Re-anchor this segment's last frame to the faceswap face before it seeds the next
    # segment. Uses faceswap_image, so it only has an effect when faceswap is configured.
    seed_faceswap: bool = False
    negative_prompt: Optional[str] = None
    auto_finalize: bool = False
    transition: Optional[str] = None
    video_preset_id: Optional[UUID] = None


# The vocabulary lives server-side so the console and any analysis agree on the exact strings.
# Grouping is the entire point of tags; two spellings of the same observation are two labels.
#
# Every tag is <location>-<condition>: where you were looking, then what was wrong or right with
# it. That keeps the chip row scannable by region and makes the set extensible without debate
# about naming -- a new observation about the eyes is eyes-<something>, not a fresh coinage.
#
OBSERVATION_TAGS = [
    # Face
    "face-frozen",
    "face-blurry",
    "face-expressive",
    "mouth-void",
    "teeth-mush",
    "identity-drift",
    # Motion, per person. motion_magnitude is whole-frame Farneback optical flow: it sums every
    # moving pixel, so a lively woman with a static man scores identically to the reverse. The
    # metric cannot separate them even in principle, which makes these the ONLY per-person
    # motion data obtainable.
    "him-static",
    "her-static",
    "him-strong",
    "her-strong",
    # Pace. Unlike every other axis, the middle value has to be explicit: with only fast/slow,
    # an untagged segment is ambiguous between "the pace was fine" and "I did not judge the
    # pace", and those are different data.
    "pace-slow",
    "pace-right",
    "pace-fast",
    # Body mechanics -- the single largest quality differentiator observed so far, and one no
    # metric can see: two bodies rocking TOGETHER generate enormous whole-frame optical flow
    # while being the wrong motion entirely. The 5-rated segment scored 0.545 and a 3-rated one
    # scored 1.162. motion_magnitude measures quantity; these record correctness.
    #
    # "rocking" is the reviewer's own word for the failure ("they are moving in unison, they
    # should be smacking together, not rocking") and it is used deliberately: the earlier name,
    # bodies-unison, described the behaviour without saying it was wrong, where every other tag
    # puts the judgement in the condition.
    "bodies-rocking",
    # The positive is needed for the same reason pace-right is: without it, an untagged segment
    # is ambiguous between "the bodies impacted properly" and "I did not judge the mechanics" --
    # and impact is the outcome round 2 is hunting for.
    "bodies-impact",
    "anatomy-break",
]

# Sets whose members contradict each other. A segment cannot be both too fast and too slow, and
# a man cannot be both static and thrusting strongly. Rejecting these server-side matters because
# the tags exist to be ground truth -- a contradictory label is worse than a missing one, since
# it quietly poisons whatever it is later used to validate.
EXCLUSIVE_TAG_GROUPS = [
    {"pace-slow", "pace-right", "pace-fast"},
    {"him-static", "him-strong"},
    {"her-static", "her-strong"},
    {"bodies-rocking", "bodies-impact"},
]


class SegmentAnnotation(BaseModel):
    """What a human saw. Never read by generation."""

    notes: Optional[str] = Field(None, max_length=4000)
    rating: Optional[int] = Field(None, ge=1, le=5, description="Overall, 1-5")
    # Sent as a list and stored comma separated; validated against OBSERVATION_TAGS so a typo
    # cannot quietly create a ninth tag that groups with nothing.
    observation_tags: Optional[list[str]] = None


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
    faceswap_model: Optional[str] = None
    faceswap_pixel_boost: Optional[str] = None
    seed_faceswap: bool = False
    negative_prompt: Optional[str] = None
    auto_finalize: bool
    transition: Optional[str]
    notes: Optional[str] = None
    rating: Optional[int] = None
    observation_tags: Optional[str] = None
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
    faceswap_model: Optional[str] = None
    faceswap_pixel_boost: Optional[str] = None
    initial_reference_image: Optional[str] = None
    # Identity scoring ground truth: the JOB's starting image, i.e. segment 0's start frame.
    # Deliberately NOT identity_reference_image - that field is the PainterLongVideo anchor
    # and is overridable, which would silently swap what "her" means mid-measurement.
    # Every segment scores against this same frame, so the numbers chain across the job.
    identity_ground_truth: Optional[str] = None
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
    notes: Optional[str] = None
    rating: Optional[int] = None
    observation_tags: Optional[str] = None
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
    faceswap_model: Optional[str] = None
    faceswap_pixel_boost: Optional[str] = None


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
