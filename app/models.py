import uuid
from datetime import datetime, timezone

from sqlalchemy import BigInteger, Boolean, DateTime, Float, ForeignKey, Index, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, mapped_column, relationship

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
    priority = mapped_column(Integer, nullable=False, default=0)
    status = mapped_column(String(20), nullable=False, default=JobStatus.PENDING)
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


class PromptPreset(Base):
    __tablename__ = "prompt_presets"

    id = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = mapped_column(String(255), unique=True, nullable=False)
    prompt = mapped_column(Text, nullable=False)
    loras = mapped_column(JSON, nullable=True)
    created_at = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
