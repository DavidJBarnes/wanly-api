import uuid
from datetime import datetime, timezone

from sqlalchemy import BigInteger, Boolean, DateTime, Float, ForeignKey, Index, Integer, JSON, String, Text, UniqueConstraint, text
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
    # Optional link to a named video-settings preset (job default). Live: the 7 sampler values
    # are read from the preset at claim time. NULL -> use this job's raw params above.
    priority = mapped_column(Integer, nullable=False, default=0)
    # Per-job continuation-mode override ("traditional"|"vace"); NULL -> global app setting.
    continuation_mode = mapped_column(String(20), nullable=True)
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
        # PARTIAL: live rows only. A discarded segment keeps its index so the record reads
        # correctly, and its replacement takes the same position in the video -- otherwise the
        # regenerated segment would have to be appended and would play out of order.
        Index("uq_segments_job_index_live", "job_id", "index",
              unique=True, postgresql_where=text("NOT discarded")),
        Index("ix_segments_job_id", "job_id"),
        Index("ix_segments_status", "status"),
    )

    id = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id = mapped_column(UUID(as_uuid=True), ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False)
    index = mapped_column(Integer, nullable=False)
    # Soft delete. The row survives with its video and its seed; the video does not include it.
    # A bad take is still the record of what that seed produced, so destroying it to get it out
    # of the cut is exactly backwards.
    discarded = mapped_column(Boolean, nullable=False, default=False, server_default="false")
    # The noise seed this segment generates with, when it has one of its own.
    #
    # NULL is the normal case and means "derive it", which is how every segment worked before
    # this column existed: the claim endpoint computes (job.seed + index), so segment 0 is exactly
    # job.seed and the whole job reproduces from that one number. Existing rows are all NULL and
    # behave exactly as they always did — this column is additive, never backfilled.
    #
    # It is set when a segment needs a seed that is NOT a function of its position, which today
    # means re-rolling segment 0 to see a different take of the same prompt. Without a per-segment
    # seed the only way to re-roll would be to overwrite job.seed, and that silently rewrites
    # history: the archived clip keeps its video while the number that produced it is replaced by
    # the number that produced its replacement. That is the worst possible thing to
    # lose here, because seed is the dominant variable in what a take actually looks like —
    # expression in particular is seed-driven far more than it is LoRA-driven — so "which seed was
    # that one?" is the question the whole archive exists to answer.
    #
    # Written in the JS-safe integer range (< 2^53) rather than Postgres' full BigInteger range.
    # A seed only has value if it can be read off the screen and used again, and JSON numbers
    # above 2^53 are silently rounded by every browser, so a larger seed would display and
    # round-trip as a DIFFERENT number than the one that generated the video.
    seed = mapped_column(BigInteger, nullable=True)
    prompt = mapped_column(Text, nullable=False)
    prompt_template = mapped_column(Text, nullable=True)
    duration_seconds = mapped_column(Float, nullable=False, default=5.0)
    speed = mapped_column(Float, nullable=False, default=1.0)
    start_image = mapped_column(Text, nullable=True)
    auto_finalize = mapped_column(Boolean, nullable=False, default=False)
    transition = mapped_column(String(20), nullable=True, default=None)
    trim_start_frames = mapped_column(Integer, nullable=False, default=0)
    trim_end_frames = mapped_column(Integer, nullable=False, default=0)
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
    negative_prompt = mapped_column(Text, nullable=True)
    # LTX recipe render: which validated (character, pose) configuration produced this
    # segment, and any of its defaults the user overrode. One JSONB rather than a column
    # per parameter, because across sixteen validated recipes every field except the
    # character LoRA and the prompt had exactly ONE distinct value — a recipe is
    # (character LoRA, prompt) and the rest is one global configuration.
    #
    # graph_sha256 is the regression trail: a recipe is value patches on a pinned graph, so
    # the hash of the resolved graph detects any change to shared state that alters a recipe,
    # at no GPU cost. It is what makes a render provably the configuration that was signed
    # off rather than one that merely claims to be.
    #
    # NULL means "not an LTX recipe render" — every WAN segment, and any free-form LTX one.
    ltx_recipe = mapped_column(JSONB, nullable=True)
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
    status = mapped_column(String(20), nullable=False, default=SegmentStatus.PENDING)
    worker_id = mapped_column(UUID(as_uuid=True), nullable=True)
    worker_name = mapped_column(String(255), nullable=True)
    # Snapshotted at claim time. Worker rows vanish when a pod deregisters, so joining to
    # workers later would lose every segment a since-terminated pod ran.
    gpu_name = mapped_column(String(100), nullable=True)
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
    # Set by workers running on RunPod. NULL for the 3090 and anything else self-hosted.
    # Pairing pods to workers by name only worked for launcher-created pods; this is the
    # identifier both sides actually agree on.
    runpod_pod_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    last_heartbeat: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    registered_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    gpu_stats: Mapped[dict | None] = mapped_column(JSONB, nullable=True, default=None)
    sd_scripts: Mapped[dict | None] = mapped_column(JSONB, nullable=True, default=None)
    a1111: Mapped[dict | None] = mapped_column(JSONB, nullable=True, default=None)
    # What this worker's LoRA directory held as of its last sync — a cached verdict with a
    # `synced_at`, not a live check: verifying one LoRA means hashing 650 MB, which a
    # heartbeat cannot do. NULL means "never reported" (or an older daemon), which is not
    # the same as an empty inventory and should not render the same way. See daemon#165.
    loras: Mapped[dict | None] = mapped_column(JSONB, nullable=True, default=None)
    # Base models this worker can load, as ComfyUI reports them. Reported rather than
    # discovered: the engine binds to localhost, so the daemon is the only thing that can
    # ask. NULL means never reported — not the same as none available. See console#404.
    checkpoints: Mapped[list | None] = mapped_column(JSONB, nullable=True, default=None)
    drain_after_jobs: Mapped[int | None] = mapped_column(Integer, nullable=True, default=None)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))


class GpuReservation(Base):
    """A standing request to launch a worker as soon as a GPU frees up.

    Persisted rather than held in memory, for two reasons that are not stylistic: the browser
    tab that created it will be closed, and the API container is recreated on every deploy. A
    reservation that dies on deploy is worse than no feature, because it dies invisibly — the
    user is still waiting for a worker that nobody is going to launch.
    """

    __tablename__ = "gpu_reservations"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # Becomes both the pod name and the worker's FRIENDLY_NAME, so the Workers page can join
    # them without a second identifier.
    name: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    # Optional, and more important here than at launch time: a reservation can fire unattended,
    # so "get me a 4090 and drain it after 3 jobs" is a bounded instruction where "get me a
    # 4090" is open-ended spend.
    drain_after_jobs = mapped_column(Integer, nullable=True)
    # Which GPU to wait for. NULL means the server default, which is what every reservation
    # created before this column existed was waiting for. It matters more than at launch time:
    # a reservation polls unattended for up to 12 hours, and a 4090 that cannot be placed would
    # burn the entire window while a 3090 would have been had in minutes.
    gpu_type_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    pod_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error: Mapped[str | None] = mapped_column(String(500), nullable=True)
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )


class LtxCharacter(Base):
    """A character: a LoRA and the strengths it runs at.

    "Adding a character costs a LoRA and a trigger swap" — that is the whole model. The
    strengths sit here rather than globally so a future character can differ, but all three
    seeded characters share 0.8/1.5.
    """
    __tablename__ = "ltx_characters"

    id = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = mapped_column(String(64), nullable=False, unique=True)
    char_lora = mapped_column(Text, nullable=False)
    # The token that fills a pose's <TRIGGER> placeholder. "Adding a character costs a LoRA
    # and a trigger swap" — this is the trigger half, and it is why a new LoRA is never
    # locked out: every pose works for it the moment the row exists.
    trigger = mapped_column(String(64), nullable=False)
    # Per-stage, never flat. Stage 1 generates at half size from noise; stage 2 refines the
    # 2x-upscaled latent and is where facial detail resolves. Collapsing them to one number
    # is a different configuration, not a simplification.
    strength_stage_1 = mapped_column(Float, nullable=False, server_default="0.8")
    strength_stage_2 = mapped_column(Float, nullable=False, server_default="1.5")
    created_at = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    # passive_deletes leaves the cascade to the database, where the FK already declares
    # ON DELETE CASCADE. Without it the ORM tries to load every child row to delete them
    # individually — which under asyncpg is a lazy load in the wrong context and raises
    # MissingGreenlet, so deleting a character failed outright. Found by running it.


class LtxRecipe(Base):
    """A POSE. Not a pose-per-character.

    Recipes are not LoRA-specific and never were: strip the leading trigger token and all 8
    seeded poses are character-agnostic. Storing them per character locked new LoRAs out —
    a character with no rows had no recipes — and let three copies of one prompt drift apart,
    which two of them already had.

    The prompt carries a <TRIGGER> placeholder that the character's own trigger word fills.
    """
    __tablename__ = "ltx_recipes"
    __table_args__ = (
        UniqueConstraint("name", name="uq_ltx_recipe_name"),
    )

    id = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = mapped_column(String(128), nullable=False)
    # Contains <TRIGGER>, filled with the character's trigger word. Substituted BEFORE
    # wildcard resolution, and the name is reserved in the wildcard routes, so the resolver
    # can never treat it as a wildcard and swap in something random.
    prompt_template = mapped_column(Text, nullable=False)
    # NULL means "the stack's negative" — true of all 24 seeded recipes. The override exists
    # because a pose might one day need one, not because any does.
    negative_prompt = mapped_column(Text, nullable=True)
    frames = mapped_column(Integer, nullable=True)
    # The POSE is proven: this prompt produces what it claims. Whether a given CHARACTER
    # renders well is a property of its LoRA, and segment ratings already record that.
    # NOT a quality score: the automated metrics have picked the wrong clip before.
    validated = mapped_column(Boolean, nullable=False, server_default="false", default=False)
    # NULL means "use the stack's value", exactly as frames and negative_prompt do.
    # A video CRF applied to the conditioning frame before it anchors the render;
    # 0 bypasses it. See wanly-api#235.
    img_compression = mapped_column(Integer, nullable=True)
    # Content LoRAs — WHAT IS HAPPENING, motion and act — chained ahead of the character
    # LoRA, which is WHO. A LIST because they stack: motion, act and framing are separable
    # and a pose may want several (console#410).
    #
    # [{"name": str, "s1": float, "s2": float}], and ORDER IS SIGNIFICANT — it is the order
    # they are applied in the chain. Two poses with the same LoRAs in a different order are
    # different configurations. Not a set.
    #
    # Empty list means none, which is what every pose did before this existed and what most
    # still do. Per-stage strengths on each because stage 1 generates at half size from
    # noise and stage 2 refines the 2x-upscaled latent; lowering a competing content LoRA on
    # stage 2 is a recorded lever and one flat number cannot express it.
    content_loras = mapped_column(JSONB, nullable=True)
    checkpoint = mapped_column(Text, nullable=True)
    created_at = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = mapped_column(DateTime(timezone=True), nullable=True)
