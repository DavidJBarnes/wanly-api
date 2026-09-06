import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from app.enums import WorkerKind, WorkerStatus


class WorkerRegister(BaseModel):
    friendly_name: str
    hostname: str
    ip_address: str
    comfyui_running: bool = False
    # Optional: only RunPod workers have one, and older daemons do not send it.
    runpod_pod_id: str | None = None
    # WHAT THIS WORKER IS (#269). Defaulted rather than required, so every daemon in the fleet
    # keeps registering unchanged and keeps meaning exactly what it meant before. A service
    # has to say so explicitly, which is the right way round: the failure of forgetting is
    # "a service is treated as a render worker", and that is caught the first time it is
    # offered a segment it cannot do -- loudly -- rather than the reverse, which is a render
    # worker silently claiming nothing.
    kind: WorkerKind = WorkerKind.RENDER
    # WHAT IT RUNS: ["ltx-engine"], ["joycaption", "qwen-edit"]. Optional, and None means
    # "never reported" rather than "runs nothing" -- the distinction every other optional
    # worker field here already draws.
    provides: list[str] | None = None


class WorkerHeartbeat(BaseModel):
    comfyui_running: bool
    runpod_pod_id: str | None = None
    gpu_stats: dict[str, Any] | None = None
    sd_scripts: dict[str, Any] | None = None
    a1111: dict[str, Any] | None = None
    # Cached LoRA inventory from the worker's last sync. Optional so an older daemon still
    # heartbeats successfully rather than 422-ing itself out of the pool on upgrade day.
    loras: dict[str, Any] | None = None
    # Base models this worker can load. Optional so an older daemon still heartbeats.
    checkpoints: list[str] | None = None
    # What code this worker is running (wanly-gpu-docker#72). Optional like everything else
    # here: a daemon that does not report them must keep heartbeating, and absent is read as
    # "does not report", never as a value.
    daemon_commit: str | None = Field(default=None, max_length=64)
    image_ref: str | None = Field(default=None, max_length=200)
    # Artifact kinds this worker can fetch on demand ("lora" today). Optional, like every
    # field before it: an older daemon that sends nothing must keep heartbeating rather than
    # 422 itself out of the pool on upgrade day. Absent is read as "fetches nothing", which
    # is the safe direction — it can still claim work whose files it already holds.
    fetchable_kinds: list[str] | None = None
    # Repeated on the heartbeat, not just at registration, because a services container can
    # change what it runs without re-registering -- SERVICES is a restart away, and the row
    # would otherwise keep advertising yesterday's set. Optional and None means "not
    # reported", so a daemon that never sends it leaves the stored value alone.
    provides: list[str] | None = None
    # A service reports its own health here: "online" when everything it was asked to run is
    # answering, "degraded" when some of it is not. Render workers do not send this -- their
    # status is derived from claims -- so None means "leave it alone".
    status: WorkerStatus | None = None


class WorkerRename(BaseModel):
    friendly_name: str


class WorkerStatusUpdate(BaseModel):
    status: str


class WorkerDrain(BaseModel):
    after_jobs: int | None = None


class QueueHealthResponse(BaseModel):
    """Work waiting versus workers able to take it.

    `stalled` requires BOTH halves: queued work with a busy worker is a queue doing its job, and
    no workers with an empty queue is a quiet night. An alarm on either alone fires constantly
    and then gets ignored.
    """

    pending_segments: int
    live_workers: int
    stalled: bool
    last_worker_seen: datetime | None = None
    # Pre-rendered so every surface phrases an outage identically.
    summary: str = ""


class WorkerResponse(BaseModel):
    id: uuid.UUID
    friendly_name: str
    hostname: str
    ip_address: str
    status: str
    # #269. `kind` is never null -- the column is NOT NULL with a default -- so the console
    # can rely on it and does not need a fallback branch for "unclassified worker".
    kind: str
    provides: list[str] | None = None
    comfyui_running: bool
    gpu_stats: dict[str, Any] | None = None
    sd_scripts: dict[str, Any] | None = None
    a1111: dict[str, Any] | None = None
    loras: dict[str, Any] | None = None
    checkpoints: list[str] | None = None
    # What the worker says it can FETCH, beside what it already holds. Stored since #249 and
    # missing from this schema until now, so it read as NULL everywhere it was looked at —
    # including while checking whether a restarted daemon had picked the field up. Pydantic
    # drops what a response model does not name, silently, which makes a stored value and an
    # unreported one indistinguishable from outside.
    fetchable_kinds: list[str] | None = None
    # Named here deliberately, and this is the point of the whole change: a stored value that
    # a response model omits is silently indistinguishable from an unreported one -- which is
    # the mistake fetchable_kinds made above, and this field exists to END that class of
    # invisibility, so it must not repeat it.
    daemon_commit: str | None = None
    image_ref: str | None = None
    drain_after_jobs: int | None = None
    last_heartbeat: datetime
    registered_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}
