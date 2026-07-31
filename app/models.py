import uuid
from datetime import datetime, timezone

from sqlalchemy import BigInteger, Boolean, DateTime, Float, ForeignKey, Index, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from app.enums import JobStatus, SegmentStatus, VideoStatus


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = mapped_column(String(255), unique=True, nullable=False)
    password_hash = mapped_column(String(255), nullable=False)
    created_at = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    jobs = relationship("Job", back_populates="user")


class Job(Base):
    __tablename__ = "jobs"
    __table_args__ = (
        Index("ix_jobs_user_id", "user_id"),
        Index("ix_jobs_status", "status"),
        Index("ix_jobs_priority", "priority"),
        Index("ix_jobs_starting_image", "starting_image"),
    )

    id = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    name = mapped_column(String(255), nullable=False)
    width = mapped_column(Integer, nullable=False)
    height = mapped_column(Integer, nullable=False)
    fps = mapped_column(Integer, nullable=False)
    seed = mapped_column(BigInteger, nullable=False)
    starting_image = mapped_column(Text, nullable=True)
    starting_image_hash = mapped_column(String(64), nullable=True, index=True)
    lightx2v_strength_high = mapped_column(Float, nullable=True)
    lightx2v_strength_low = mapped_column(Float, nullable=True)
    cfg_high = mapped_column(Float, nullable=True)
    cfg_low = mapped_column(Float, nullable=True)
    steps_total = mapped_column(Integer, nullable=True)
    high_noise_steps = mapped_column(Integer, nullable=True)
    flow_shift = mapped_column(Float, nullable=True)
    # Optional link to a named video-settings preset (job default). Live: the 7 sampler values
    # are read from the preset at claim time. NULL -> use this job's raw params above.
    video_preset_id = mapped_column(
        UUID(as_uuid=True), ForeignKey("video_settings_presets.id", ondelete="SET NULL"), nullable=True
    )
    priority = mapped_column(Integer, nullable=False, default=0)
    config_starred = mapped_column(Boolean, nullable=False, default=False)
    # Per-job continuation-mode override ("traditional"|"vace"); NULL -> global app setting.
    continuation_mode = mapped_column(String(20), nullable=True)
    # Canonical identity reference for VACE continuation: a face crop from seg0, set by the daemon
    # after seg0 completes and fed to every downstream segment's VACE ref_images (anchors identity).
    identity_reference_image = mapped_column(Text, nullable=True)
    # === Lynx identity-preserving engine (ByteDance Lynx on Wan2.1 T2V-14B) ===
    # generation_engine selects the daemon's graph builder: NULL/"wan22" -> the default
    # 2.2 i2v path, "lynx" -> build_lynx_workflow. Lynx is a different base model family,
    # so the daemon fails loudly rather than falling back if it cannot run it.
    generation_engine = mapped_column(String(20), nullable=True)
    # Subject image conditioning identity via ArcFace + VAE reference features. NOT a start
    # frame — this is a T2V base, so the subject never appears as frame 0.
    lynx_subject_image = mapped_column(Text, nullable=True)
    # Adapter strengths: ip = who the face is (ID adapter), ref = fine appearance detail
    # (reference adapter). NULL -> the daemon's settings default.
    lynx_ip_scale = mapped_column(Float, nullable=True)
    lynx_ref_scale = mapped_column(Float, nullable=True)
    lynx_cfg_scale = mapped_column(Float, nullable=True)
    # Denoise window over which the ref adapter applies, as a fraction of total steps.
    lynx_start_percent = mapped_column(Float, nullable=True)
    lynx_end_percent = mapped_column(Float, nullable=True)
    # Comma-separated DiT block indices/ranges for the ref feature; NULL/"" -> all blocks.
    lynx_ref_blocks_to_use = mapped_column(Text, nullable=True)
    # A/B arm. These are a MATCHED PAIR — a mixed pair loads silently and yields garbage
    # identity, so the daemon rejects a mismatch.
    lynx_ip_layers = mapped_column(Text, nullable=True)
    lynx_resampler = mapped_column(Text, nullable=True)
    lynx_steps = mapped_column(Integer, nullable=True)
    lynx_cfg = mapped_column(Float, nullable=True)
    lynx_shift = mapped_column(Float, nullable=True)
    lynx_scheduler = mapped_column(String(32), nullable=True)
    lynx_distill_strength = mapped_column(Float, nullable=True)
    status = mapped_column(String(20), nullable=False, default=JobStatus.PENDING)
    tags = mapped_column(Text, nullable=True)
    created_at = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    user = relationship("User", back_populates="jobs")
    segments = relationship("Segment", back_populates="job", order_by="Segment.index", cascade="all, delete-orphan", passive_deletes=True)
    videos = relationship("Video", back_populates="job", cascade="all, delete-orphan", passive_deletes=True)


class Segment(Base):
    __tablename__ = "segments"
    __table_args__ = (
        UniqueConstraint("job_id", "index", name="uq_segments_job_index"),
        Index("ix_segments_job_id", "job_id"),
        Index("ix_segments_status", "status"),
    )

    id = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id = mapped_column(UUID(as_uuid=True), ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False)
    index = mapped_column(Integer, nullable=False)
    prompt = mapped_column(Text, nullable=False)
    prompt_template = mapped_column(Text, nullable=True)
    duration_seconds = mapped_column(Float, nullable=False, default=5.0)
    speed = mapped_column(Float, nullable=False, default=1.0)
    start_image = mapped_column(Text, nullable=True)
    loras = mapped_column(JSON, nullable=True)
    faceswap_enabled = mapped_column(Boolean, nullable=False, default=False)
    faceswap_method = mapped_column(String(20), nullable=True)
    faceswap_source_type = mapped_column(String(20), nullable=True)
    faceswap_image = mapped_column(Text, nullable=True)
    faceswap_faces_order = mapped_column(Text, nullable=True)
    faceswap_faces_index = mapped_column(Text, nullable=True)
    auto_finalize = mapped_column(Boolean, nullable=False, default=False)
    transition = mapped_column(String(20), nullable=True, default=None)
    trim_start_frames = mapped_column(Integer, nullable=False, default=0)
    trim_end_frames = mapped_column(Integer, nullable=False, default=0)
    motion_keywords = mapped_column(JSON, nullable=True)
    motion_magnitude = mapped_column(Float, nullable=True)
    # Length (seconds) of the reconstructed lead-in a VACE-continuation segment carries.
    # Stitch trims this off the previous segment's tail so the reconstruction replaces it
    # seamlessly. NULL for traditional (non-VACE) segments.
    vace_overlap_seconds = mapped_column(Float, nullable=True)
    # AR hologram (Tier-0). When a finalized job's index-0 segment is reused as the carrier
    # for reprocess_type="ar_hologram": the two params drive the daemon matte + manifest, the
    # three paths hold the packed color+alpha mp4, the hologram.json manifest, and the poster.
    hologram_key_color = mapped_column(String(20), nullable=True)
    hologram_subject_height_m = mapped_column(Float, nullable=True)
    # "2d_matte" (flat, Tier-0) or "2.5d_depth" (depth-displaced mesh, Tier-1). One flavor per
    # video at a time — re-making overwrites the single carrier's artifacts.
    hologram_flavor = mapped_column(String(16), nullable=True)
    # Relief depth in meters for the 2.5d_depth flavor (how far the nearest pixels are pushed
    # toward the viewer). Per-remake knob from the console dialog; daemon falls back to its
    # config default when null.
    hologram_depth_scale_m = mapped_column(Float, nullable=True)
    hologram_video_path = mapped_column(Text, nullable=True)
    hologram_manifest_path = mapped_column(Text, nullable=True)
    hologram_poster_path = mapped_column(Text, nullable=True)
    reference_frames = mapped_column(JSON, nullable=True)
    # Lynx identity QA, written by the daemon after a Lynx render: per-frame cosine
    # similarities of sampled output frames against the subject, plus summary stats.
    # Measurement only — nothing gates on it. Shape:
    # {"scores": [...], "mean": f, "min": f, "max": f,
    #  "frames_sampled": n, "frames_with_face": n}
    lynx_identity_scores = mapped_column(JSON, nullable=True)
    negative_prompt = mapped_column(Text, nullable=True)
    reprocess_type = mapped_column(String(20), nullable=True)
    # Foundry smashcut carrier (reprocess_type="smashcut_concat"): ordered list of source clip
    # output_paths to concatenate + the transition style ("seamless" | "black").
    smashcut_clip_paths = mapped_column(JSON, nullable=True)
    smashcut_transition = mapped_column(String(20), nullable=True)
    # Per-clip playback speed, aligned 1:1 with smashcut_clip_paths. NULL means "no retiming"
    # (the common case) and keeps the daemon on its fast stream-copy concat path. Distinct
    # from Segment.speed above, which is a generation-time motion-density knob.
    smashcut_clip_speeds = mapped_column(JSON, nullable=True)
    # Per-segment video-settings override (live link). Takes precedence over the job's preset.
    video_preset_id = mapped_column(
        UUID(as_uuid=True), ForeignKey("video_settings_presets.id", ondelete="SET NULL"), nullable=True
    )
    status = mapped_column(String(20), nullable=False, default=SegmentStatus.PENDING)
    worker_id = mapped_column(UUID(as_uuid=True), nullable=True)
    worker_name = mapped_column(String(255), nullable=True)
    output_path = mapped_column(Text, nullable=True)
    last_frame_path = mapped_column(Text, nullable=True)
    created_at = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    claimed_at = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at = mapped_column(DateTime(timezone=True), nullable=True)
    error_message = mapped_column(Text, nullable=True)
    progress_log = mapped_column(Text, nullable=True)

    job = relationship("Job", back_populates="segments")


class Video(Base):
    __tablename__ = "videos"
    __table_args__ = (
        Index("ix_videos_job_id", "job_id"),
    )

    id = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id = mapped_column(UUID(as_uuid=True), ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False)
    output_path = mapped_column(Text, nullable=True)
    duration_seconds = mapped_column(Float, nullable=True)
    status = mapped_column(String(20), nullable=False, default=VideoStatus.PENDING)
    error_message = mapped_column(Text, nullable=True)
    tags = mapped_column(Text, nullable=True)
    created_at = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    completed_at = mapped_column(DateTime(timezone=True), nullable=True)

    job = relationship("Job", back_populates="videos")


class Lora(Base):
    __tablename__ = "loras"

    id = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = mapped_column(String(255), nullable=False)
    description = mapped_column(Text, nullable=True)
    trigger_words = mapped_column(Text, nullable=True)
    default_prompt = mapped_column(Text, nullable=True)
    source_url = mapped_column(Text, nullable=True)
    preview_image = mapped_column(Text, nullable=True)
    high_file = mapped_column(String(255), nullable=True)
    high_s3_uri = mapped_column(Text, nullable=True)
    low_file = mapped_column(String(255), nullable=True)
    low_s3_uri = mapped_column(Text, nullable=True)
    default_high_weight = mapped_column(Float, nullable=False, default=1.0)
    default_low_weight = mapped_column(Float, nullable=False, default=1.0)
    created_at = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))


class TitleTag(Base):
    __tablename__ = "title_tags"
    __table_args__ = (
        Index("ix_title_tags_group", "group"),
        UniqueConstraint("name", "group", name="uq_title_tags_name_group"),
    )

    id = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = mapped_column(String(255), nullable=False)
    group = mapped_column(Integer, nullable=False)
    created_at = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))


class Wildcard(Base):
    __tablename__ = "wildcards"

    id = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = mapped_column(String(255), unique=True, nullable=False)
    options = mapped_column(JSON, nullable=False, default=list)
    created_at = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))


class AppSetting(Base):
    __tablename__ = "app_settings"

    key = mapped_column(String(255), primary_key=True)
    value = mapped_column(Text, nullable=False)
    updated_at = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))


class VideoSettingsPreset(Base):
    """A named bundle of the 7 sampler params, selectable per-job and per-segment (live link)."""

    __tablename__ = "video_settings_presets"

    id = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = mapped_column(String(255), unique=True, nullable=False)
    lightx2v_strength_high = mapped_column(Float, nullable=True)
    lightx2v_strength_low = mapped_column(Float, nullable=True)
    cfg_high = mapped_column(Float, nullable=True)
    cfg_low = mapped_column(Float, nullable=True)
    steps_total = mapped_column(Integer, nullable=True)
    high_noise_steps = mapped_column(Integer, nullable=True)
    flow_shift = mapped_column(Float, nullable=True)
    # Sampler algorithm + scheduler (NULL -> daemon default euler/simple). Only the scheduler
    # applies on the VACE path (its sampler node has no sampler_name).
    sampler_name = mapped_column(String(40), nullable=True)
    scheduler = mapped_column(String(40), nullable=True)
    # 1:N LoRAs that constitute this recipe — each {lora_id, high_weight, low_weight} (expert
    # placement). Resolved live at claim time when a job/segment links this preset.
    loras = mapped_column(JSON, nullable=True)
    # Default prompt for this recipe. A snapshot default that fills the prompt field at job
    # creation (overridable at submit) — NOT live-linked like loras/sampler params.
    prompt = mapped_column(Text, nullable=True)
    # Free-form notes about this recipe (what it's for, gotchas). Not used by generation.
    notes = mapped_column(Text, nullable=True)
    created_at = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))


class Favorite(Base):
    __tablename__ = "favorites"
    __table_args__ = (
        UniqueConstraint("user_id", "item_type", "item_ref", name="uq_favorites_user_type_ref"),
        Index("ix_favorites_user_type", "user_id", "item_type"),
    )

    id = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    item_type = mapped_column(String(20), nullable=False)
    item_ref = mapped_column(String(500), nullable=False)
    created_at = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class ImageMeta(Base):
    __tablename__ = "image_meta"

    path = mapped_column(Text, primary_key=True)
    tags = mapped_column(Text, nullable=True)
    updated_at = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))


class Worker(Base):
    __tablename__ = "workers"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    friendly_name: Mapped[str] = mapped_column(String(255), nullable=False)
    hostname: Mapped[str] = mapped_column(String(255), nullable=False)
    ip_address: Mapped[str] = mapped_column(String(45), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="online-idle")
    comfyui_running: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    last_heartbeat: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    registered_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    gpu_stats: Mapped[dict | None] = mapped_column(JSONB, nullable=True, default=None)
    sd_scripts: Mapped[dict | None] = mapped_column(JSONB, nullable=True, default=None)
    a1111: Mapped[dict | None] = mapped_column(JSONB, nullable=True, default=None)
    drain_after_jobs: Mapped[int | None] = mapped_column(Integer, nullable=True, default=None)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
