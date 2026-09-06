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


class WorkerKind(StrEnum):
    """What a worker IS — the only thing the claim gate reads (#269).

    Deliberately a tiny closed set rather than the list of things a worker runs. A gate keyed
    on names needs an allowlist, and an engine missing from that allowlist claims nothing,
    which is indistinguishable from an empty queue. Here a new engine is RENDER and claimable
    by default, while a new service is SERVICE and excluded by default: both defaults land on
    the safe side.

    What a worker runs is `Worker.provides`, which is a list and is for display.
    """
    RENDER = "render"
    SERVICE = "service"


#: Convenience for the model's column defaults, which cannot reference the enum member
#: directly without dragging StrEnum into the DDL.
WORKER_KIND_RENDER = str(WorkerKind.RENDER)


class WorkerStatus(StrEnum):
    """Worker statuses, across both kinds.

    ONLINE_IDLE and ONLINE_BUSY are the render vocabulary and mean what they always have.

    ONLINE and DEGRADED are for services, and exist because reusing ONLINE_IDLE would have
    been the cheap wrong answer: wanly-gpu-docker#80 is open precisely because online-idle
    already means two different things -- "waiting for work" and "cannot do work" -- and that
    ambiguity hid a dead ComfyUI for 33 minutes. A service is never waiting for work, so
    saying it is idle would extend exactly the confusion that is already costing us.

    DEGRADED is not a synonym for broken: wanly-services reports it when some but not all of
    its enabled services answer, which is a state a single green dot cannot express.
    """
    ONLINE_IDLE = "online-idle"
    ONLINE_BUSY = "online-busy"
    ONLINE = "online"
    DEGRADED = "degraded"
    DRAINING = "draining"
    OFFLINE = "offline"
