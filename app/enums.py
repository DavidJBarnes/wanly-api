"""Status enums for jobs, segments, and videos.

These enforce valid status values at the application layer. Database CHECK
constraints should be added via Alembic migration for production enforcement.
"""

from enum import StrEnum


class JobStatus(StrEnum):
    """Valid job statuses."""
    PENDING = "pending"
    PROCESSING = "processing"
    AWAITING = "awaiting"
    FAILED = "failed"
    PAUSED = "paused"
    FINALIZED = "finalized"
    FINALIZING = "finalizing"
    ARCHIVED = "archived"


class SegmentStatus(StrEnum):
    """Valid segment statuses."""
    PENDING = "pending"
    CLAIMED = "claimed"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class VideoStatus(StrEnum):
    """Valid video statuses."""
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"


# User-initiated job status transitions (used by PATCH /jobs/{id})
JOB_VALID_TRANSITIONS: dict[str, set[str]] = {
    JobStatus.PENDING: {JobStatus.PAUSED, JobStatus.ARCHIVED},
    JobStatus.PROCESSING: {JobStatus.PAUSED},
    JobStatus.AWAITING: {JobStatus.PAUSED, JobStatus.FINALIZED, JobStatus.ARCHIVED},
    JobStatus.FAILED: {JobStatus.PAUSED, JobStatus.ARCHIVED},
    JobStatus.PAUSED: {JobStatus.PENDING, JobStatus.PROCESSING, JobStatus.AWAITING, JobStatus.ARCHIVED},
    JobStatus.ARCHIVED: {JobStatus.AWAITING},
}
