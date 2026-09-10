import uuid
from datetime import datetime, timezone

from sqlalchemy import BigInteger, Boolean, DateTime, Float, ForeignKey, Index, Integer, JSON, String, Text, UniqueConstraint, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from app.enums import JobStatus, SegmentStatus, TrainingStatus, VideoStatus, WORKER_KIND_RENDER


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
    """What we know about an image in the repo, beyond what S3 itself stores.

    One row per s3:// path, created on first use. A row exists only because something was
    said about the image — tags, or a description — so the absence of a row is meaningful:
    `list_untagged_images` reads it as "never tagged".
    """
    __tablename__ = "image_meta"

    path = mapped_column(Text, primary_key=True)
    tags = mapped_column(Text, nullable=True)
    # JoyCaption's description of this frame, produced once and reused (console#414).
    #
    # Cached rather than re-derived because it costs 2070 time — 4.5s cold, 1.2s warm — and
    # the same image starts many jobs. It is not a derived value that can be recomputed for
    # free: the model is nondeterministic, so a second call gives DIFFERENT words, and the
    # words are what the person read and accepted.
    scene_description = mapped_column(Text, nullable=True)
    # Which instruction produced it. A caption written under "terse" and one under "rich"
    # are different artefacts, and the row has to say which one this is — the same reason
    # CaptionResponse returns it.
    scene_instruction = mapped_column(Text, nullable=True)
    scene_described_at = mapped_column(DateTime(timezone=True), nullable=True)
    updated_at = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    def is_empty(self) -> bool:
        """Nothing left worth a row.

        The tags endpoint deletes a row when its tags are cleared. Once a description can
        also live here that is no longer the same question, and getting it wrong throws away
        GPU work over a tag edit.
        """
        return not (self.tags or "").strip() and not (self.scene_description or "").strip()


class Worker(Base):
    __tablename__ = "workers"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    friendly_name: Mapped[str] = mapped_column(String(255), nullable=False)
    hostname: Mapped[str] = mapped_column(String(255), nullable=False)
    ip_address: Mapped[str] = mapped_column(String(45), nullable=False)
    # WHAT THIS WORKER IS, read by the claim gate and by queue-health (#269). `render` takes
    # segments; `service` never can. Scalar and NOT NULL on purpose: there must be no window
    # in which a live worker's claimability is unknown, because the gate would then have to
    # guess, and the safe guess and the useful guess are opposites.
    #
    # Defaulting to `render` keeps every existing row and every current daemon meaning exactly
    # what it meant before this column existed.
    # Every kind this worker is (wanly-gpu-docker#83): a box running the render stack and the
    # trainer registers once as ["render", "trainer"]. `kind` below is the first of these,
    # with render first whenever present, so the render gates and the console keep reading
    # one word. Nullable: rows from before the column are read as [kind].
    kinds: Mapped[list | None] = mapped_column(JSONB, nullable=True, default=None)
    kind: Mapped[str] = mapped_column(String(20), nullable=False, default=WORKER_KIND_RENDER,
                                      server_default=WORKER_KIND_RENDER)
    # WHAT IT RUNS, for display: ["ltx-engine"], ["joycaption", "qwen-edit"]. A list because a
    # services container runs several at once, which is the entire point of its SERVICES flag.
    #
    # Deliberately NOT what the gate reads. A gate keyed on names needs an allowlist, and a new
    # engine missing from that allowlist claims nothing -- indistinguishable from an empty
    # queue, which is the failure this codebase keeps paying for.
    #
    # NULL means never reported, [] means reports nothing. Same distinction as checkpoints and
    # loras above, same reason: an older daemon must not look like one with nothing to offer.
    provides: Mapped[list | None] = mapped_column(JSONB, nullable=True, default=None)
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
    # Artifact kinds this worker can FETCH on demand, as it reports them: today ["lora"],
    # because the daemon downloads a LoRA a pose names but this worker has never seen. It is
    # declared by the worker rather than assumed by the API so that a daemon which learns to
    # fetch checkpoints (console#423) opens the claim gate by saying so — no API change, and
    # no second opinion here about what a daemon can do. NULL means never reported, which
    # this treats as fetching nothing. See app/model_requirements.py.
    fetchable_kinds: Mapped[list | None] = mapped_column(JSONB, nullable=True, default=None)
    # What code this worker is actually running (wanly-gpu-docker#72). TWO fields, because
    # there are two update channels that drift separately: the daemon is re-cloned from main
    # at every container boot, while start.sh, the downloader and the engine only change on
    # an image pull + recreate. A `docker restart` moves the first and not the second, which
    # is exactly how the 3090 came to run a current daemon on a 37-hour-old image.
    daemon_commit: Mapped[str | None] = mapped_column(Text, nullable=True, default=None)
    image_ref: Mapped[str | None] = mapped_column(Text, nullable=True, default=None)
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


class Dataset(Base):
    """A named, taggable set of images kept for training.

    THE THING THAT DID NOT EXIST. Until now the only groupings were an S3 folder prefix and a
    per-user favourites list, so "these 27 images are p@y v2's training set" could not be
    written down -- you re-selected them by hand every time, and a v3 meant doing it again from
    memory.

    Images are an explicit ORDERED list of s3:// URIs rather than a folder listing. A dataset
    usually corresponds to a folder, and uploads go into one, but the list is what the dataset
    IS: it survives an image being moved, and it records the order the trainer will stage them
    in, which pairs with the captions.
    """
    __tablename__ = "datasets"
    __table_args__ = (Index("ix_datasets_name", "name", unique=True),)

    id = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    name = mapped_column(String(100), nullable=False)
    # Comma-separated, matching ImageMeta.tags rather than inventing a second convention. The
    # console already knows how to parse, render and filter that shape.
    tags = mapped_column(String(500), nullable=True)
    notes = mapped_column(Text, nullable=True)
    #: s3:// URIs, order significant.
    images = mapped_column(JSONB, nullable=False, default=list)
    #: The S3 prefix uploads land in. Recorded rather than derived, because a rename must not
    #: silently point a dataset at a folder that does not exist.
    prefix = mapped_column(String(200), nullable=True)
    #: ONE IMAGE IN THIS SET, NOMINATED AS THE FACE EVERYTHING ELSE IS SCORED AGAINST.
    #:
    #: The alternative already here -- score against a reference DATASET's mean -- is unusable
    #: on the sets people actually have. A mean over a set that still contains two people is a
    #: blend of both and separates neither, and with no reference at all the crops are scored
    #: against their own mean, which only shows they resemble each other.
    #:
    #: One picked face has no such ambiguity: "is this the same person as THAT" is the question
    #: worth asking, and it is answerable from the moment the crops exist.
    #:
    #: A URI, not an index. The list is reordered by removal, so an index would silently come
    #: to mean a different photograph.
    anchor_uri = mapped_column(Text, nullable=True)
    created_at = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc),
                               onupdate=lambda: datetime.now(timezone.utc))


class TrainingJob(Base):
    """A character-LoRA training run: what to train, who is training it, and what came out.

    SHAPED ON Segment, deliberately. It is the same problem — a row the console creates, a
    remote worker claims, works on for a long time, and reports back about — and the existing
    machinery for that (orphan reclaim, the heartbeat sweep, the console's status rendering)
    only transfers if the shape does.

    It is a separate table rather than a Segment variant because almost nothing overlaps: no
    job, no index, no seed, no video. The two share a lifecycle, not a payload.
    """
    __tablename__ = "training_jobs"
    __table_args__ = (
        Index("ix_training_jobs_status", "status"),
        # One live run per character+version. A second attempt at v2 while the first is still
        # going would train two LoRAs into the same output name and the second would win
        # silently -- the same class of collision that made new_character.sh mix versions.
        Index("uq_training_jobs_character_version_live", "character", "version",
              unique=True,
              postgresql_where=text("status NOT IN ('completed','failed','cancelled')")),
    )

    id = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)

    # ---- what to train
    character = mapped_column(String(64), nullable=False)
    # The token the captions use and a prompt types. Allowed to differ from `character`, and on
    # p@y it must: the LoRA installs as `pay_...` because it is served over HTTP and lands in
    # JSON and URLs, while the trigger stays `p@y` because that is what trained.
    trigger = mapped_column(String(64), nullable=False)
    version = mapped_column(Integer, nullable=False, default=1)
    # The dataset, as s3:// URIs, ORDER SIGNIFICANT -- the trainer stages them as sel_000..N and
    # the captions pair by index. Precedent: Segment.smashcut_clip_paths.
    dataset_images = mapped_column(JSONB, nullable=False)
    # Everything the run needs, snapshotted. Same rule as Segment.ltx_recipe: the worker must
    # never look its configuration up for itself, because a worker that cannot look one up
    # cannot look up a STALE one. Holds caption(s), steps, rank, LR -- the recipe as it was
    # when the job was created, so a later change to the defaults cannot retroactively alter
    # what a queued job will do.
    config = mapped_column(JSONB, nullable=False, default=dict)

    # ---- who is doing it
    status = mapped_column(String(20), nullable=False, default=TrainingStatus.PENDING)
    # No FK on worker_id, matching Segment: worker rows are deleted on deregister, and a
    # finished training run should still say which box produced it.
    worker_id = mapped_column(UUID(as_uuid=True), nullable=True)
    worker_name = mapped_column(String(255), nullable=True)
    # Snapshotted at claim, because the workers row vanishes when a pod drains and "which GPU
    # trained this" is exactly the question you ask six weeks later.
    gpu_name = mapped_column(String(100), nullable=True)

    # ---- what happened
    # Free text, wholly overwritten on each report, exactly like Segment.progress_log -- and
    # load-bearing for the same reason: an empty progress log is the only trustworthy evidence
    # that a claim is not actually being worked on. Status can lie; this cannot.
    progress_log = mapped_column(Text, nullable=True)
    # Structured progress, which Segment has no equivalent of. A training run genuinely has a
    # step count, and the console already has a determinate progress bar looking for exactly
    # these two numbers.
    step = mapped_column(Integer, nullable=True)
    total_steps = mapped_column(Integer, nullable=True)
    error_message = mapped_column(Text, nullable=True)
    # Every epoch checkpoint the run produced, as reported. The choice of WHICH one to install
    # is a human judgement made by eye at a fixed seed -- loss does not rank them -- so all of
    # them are recorded rather than just the last.
    checkpoints = mapped_column(JSONB, nullable=True)
    # The s3:// URI of the installed LoRA, once one is chosen.
    output_lora_path = mapped_column(Text, nullable=True)
    # [[step, avr_loss], ...] sampled from the trainer's progress bar every report. Small --
    # a report every 20 s over an hour is under two hundred points -- and it is what lets
    # the console draw the curve rather than print the last number.
    loss_log = mapped_column(JSONB, nullable=True)
    # Every checkpoint the run WROTE, as [{label, step, loss}], published or not. Only the
    # final one is uploaded by default: a 650 MB checkpoint takes ~18 minutes to leave the
    # 3090, and "I often only want 1 or 2 epochs". The rest sit on the trainer until asked
    # for through publish_requests, and `checkpoints` records the ones that arrived.
    epochs = mapped_column(JSONB, nullable=True)
    publish_requests = mapped_column(JSONB, nullable=True)
    # Free-form operator notes, exactly as Dataset.notes: what was learned about this run,
    # why a checkpoint was picked. Written only by a human through the console's own route --
    # not through the trainer's PATCH, whose conditional writes must never be able to clobber
    # a date an editor typed.
    notes = mapped_column(Text, nullable=True)
    # The dataset's anchor image at creation -- the face this LoRA is of, for the console
    # and for the character row it publishes to.
    thumbnail_uri = mapped_column(Text, nullable=True)

    created_at = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    claimed_at = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at = mapped_column(DateTime(timezone=True), nullable=True)


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
    # The other half of the caption this LoRA trained on. Every run captions its images
    # "<trigger>, <gender>", so the identity is bound to the PAIR; filling <TRIGGER> with
    # the trigger alone left the binding word out of every render prompt, which is what
    # decides who is who when two identity LoRAs share the weights (wanly-console#487).
    # NULL for a character that predates the trainer: it renders the bare trigger as before.
    gender = mapped_column(String(16), nullable=True)
    # Per-stage, never flat. Stage 1 generates at half size from noise; stage 2 refines the
    # 2x-upscaled latent and is where facial detail resolves. Collapsing them to one number
    # is a different configuration, not a simplification.
    strength_stage_1 = mapped_column(Float, nullable=False, server_default="0.8")
    strength_stage_2 = mapped_column(Float, nullable=False, server_default="1.5")
    # A face for the LoRA: the anchor image of the dataset that trained it. Set when a
    # training run publishes to this character, editable like everything else here.
    image_uri = mapped_column(Text, nullable=True)
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
    # renders well is a property of its LoRA, which is a separate question from this flag.
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
