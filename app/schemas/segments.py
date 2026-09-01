from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, model_validator


class SegmentCreate(BaseModel):
    prompt: str
    duration_seconds: float = 5.0
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    start_image: Optional[str] = None
    negative_prompt: Optional[str] = None
    # LTX recipe render: which validated (character, pose) configuration this segment ran,
    # plus any of its defaults the user overrode and the resolved graph hash. NULL means
    # "not a recipe render". See Segment.ltx_recipe for why this is one field and not a dozen.
    ltx_recipe: Optional[dict[str, Any]] = None
    auto_finalize: bool = False
    transition: Optional[str] = None


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
    # Repetitive rather than absent: the same open-close cycle looping for the whole segment,
    # "goldfish mouth". Distinct from face-frozen (no movement) and mouth-void (the gaping black
    # space), and worth its own label because it inflates the expression metric rather than
    # depressing it -- sustained large landmark displacement is exactly what that metric rewards,
    # which makes this the third artifact it scores as a success.
    "mouth-looping",
    "teeth-mush",
    "identity-drift",
    # Motion, per person. motion_magnitude WAS whole-frame Farneback optical flow (removed in
    # #151, and this is part of why): it summed every
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
    # metric can see: two bodies moving as ONE UNIT generate enormous whole-frame optical flow
    # while being the wrong motion entirely. The 5-rated segment scored 0.545 and a 3-rated one
    # scored 1.162. The metric measured quantity; these record correctness, which is why
    # they outlived it.
    #
    # The axis is TRAVEL: he withdraws and re-enters, so the bodies separate and rejoin along an
    # axis. What fails is not that they "rock" -- that is only how the failure looks from outside
    # -- it is that they stay joined and there is no relative displacement between them. Naming
    # both ends for travel keeps the judgement on the thing being judged.
    "bodies-locked",
    "bodies-inout",
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
    {"bodies-locked", "bodies-inout"},
]


class SegmentAnnotation(BaseModel):
    """What a human saw. Never read by generation."""

    notes: Optional[str] = Field(None, max_length=4000)
    rating: Optional[int] = Field(None, ge=1, le=5, description="Overall, 1-5")
    # Sent as a list and stored comma separated; validated against OBSERVATION_TAGS so a typo
    # cannot quietly create a ninth tag that groups with nothing.
    observation_tags: Optional[list[str]] = None


class SegmentPromptUpdate(BaseModel):
    """Change what a segment will generate, before it starts.

    The prompt is a SNAPSHOT taken at job creation -- unlike loras and sampler settings, which
    resolve live from the preset at claim time. That asymmetry is deliberate (a queued job should
    not silently change what it depicts) but it left no way to correct a queued batch after
    improving the preset, short of deleting and recreating every job.
    """

    # Either supply the text directly...
    prompt: Optional[str] = Field(None, min_length=1, max_length=8000)
    # ...or re-take the snapshot from the segment's linked preset, which is the common case after
    # editing that preset.
    from_preset: bool = False

    @model_validator(mode="after")
    def exactly_one_source(self):
        if bool(self.prompt) == self.from_preset:
            raise ValueError("Supply either prompt or from_preset=true, not both or neither")
        return self


class SegmentResponse(BaseModel):
    id: UUID
    job_id: UUID
    index: int
    # A STRING, not a number, and the type is the point.
    #
    # Seeds are identifiers: their only use is being read off the screen and used again. They are
    # stored as BigInteger and 95% of existing jobs (1,645 of 1,724, measured 2026-08-14) have a
    # seed above 2**53, which is the largest integer JSON survives intact — every browser silently
    # rounds anything larger. Sent as a number, the seed displayed beside a clip would not be the
    # seed that generated it, and nothing anywhere would report a problem. Sent as a string it is
    # exact for every value, and no client does arithmetic on it anyway.
    #
    # NULL means "derived from the job" (job.seed + index) — a segment that never asked for a
    # particular seed. Archiving a take stamps the derived value in, so anything in the archive
    # carries its own answer.
    seed: Optional[str] = None

    @field_validator("seed", mode="before")
    @classmethod
    def _seed_to_string(cls, v: object) -> Optional[str]:
        """The column is an integer; the wire format is not."""
        return None if v is None else str(v)
    prompt: str
    prompt_template: Optional[str]
    duration_seconds: float
    speed: float
    start_image: Optional[str]
    negative_prompt: Optional[str] = None
    # LTX recipe render: which validated (character, pose) configuration this segment ran,
    # plus any of its defaults the user overrode and the resolved graph hash. NULL means
    # "not a recipe render". See Segment.ltx_recipe for why this is one field and not a dozen.
    ltx_recipe: Optional[dict[str, Any]] = None
    auto_finalize: bool
    transition: Optional[str]
    # Soft-deleted: kept for its rating, tags and notes, excluded from the video.
    discarded: bool = False
    notes: Optional[str] = None
    rating: Optional[int] = None
    observation_tags: Optional[str] = None
    trim_start_frames: int
    trim_end_frames: int
    reference_frames: Optional[list[str]] = None
    status: str
    reprocess_type: Optional[str] = None
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
    reference_frames: Optional[list[str]] = None
    negative_prompt: Optional[str] = None
    # LTX recipe render: which validated (character, pose) configuration this segment ran,
    # plus any of its defaults the user overrode and the resolved graph hash. NULL means
    # "not a recipe render". See Segment.ltx_recipe for why this is one field and not a dozen.
    ltx_recipe: Optional[dict[str, Any]] = None
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
    favorite: bool = False

    model_config = {"from_attributes": True}


class SegmentTrimUpdate(BaseModel):
    # Soft-deleted: kept for its rating, tags and notes, excluded from the video.
    discarded: bool = False
    notes: Optional[str] = None
    rating: Optional[int] = None
    observation_tags: Optional[str] = None
    trim_start_frames: int = Field(ge=0)
    trim_end_frames: int = Field(ge=0)


class FramePreview(BaseModel):
    frame_index: int
    data_url: str


class FramePreviewResponse(BaseModel):
    total_frames: int
    fps: float
    frames: list[FramePreview]


class HologramRequest(BaseModel):
    """Per-request overrides for a 'Make Hologram' action. Unset -> AppSetting -> hardcoded default."""

    subject_height_m: Optional[float] = None
    key_color: Optional[str] = None
    flavor: Optional[str] = None  # "2d_matte" (default) or "2.5d_depth"
    depth_scale_m: Optional[float] = None  # 2.5d relief depth in meters (clamped 0.03..0.60)


class RerollRequest(BaseModel):
    """Body for POST /jobs/{id}/reroll.

    Carried a "re-roll until" rule — a metric and a threshold, judged on completion. The
    metrics it judged are gone (#151), so a rule would be permanently unevaluable and the
    fields with it. Kept as a class because the endpoint still accepts an optional body.
    """


class SegmentStatusUpdate(BaseModel):
    status: Optional[str] = None
    output_path: Optional[str] = None
    last_frame_path: Optional[str] = None
    error_message: Optional[str] = None
    progress_log: Optional[str] = None
    vace_overlap_seconds: Optional[float] = None
    # Lynx identity QA measured by the daemon. Measurement only — no gating.
