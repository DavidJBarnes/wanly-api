import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel


class WorkerRegister(BaseModel):
    friendly_name: str
    hostname: str
    ip_address: str
    comfyui_running: bool = False
    # Optional: only RunPod workers have one, and older daemons do not send it.
    runpod_pod_id: str | None = None


class WorkerHeartbeat(BaseModel):
    comfyui_running: bool
    runpod_pod_id: str | None = None
    gpu_stats: dict[str, Any] | None = None
    sd_scripts: dict[str, Any] | None = None
    a1111: dict[str, Any] | None = None
    # Cached LoRA inventory from the worker's last sync. Optional so an older daemon still
    # heartbeats successfully rather than 422-ing itself out of the pool on upgrade day.
    loras: dict[str, Any] | None = None


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
    comfyui_running: bool
    gpu_stats: dict[str, Any] | None = None
    sd_scripts: dict[str, Any] | None = None
    a1111: dict[str, Any] | None = None
    loras: dict[str, Any] | None = None
    drain_after_jobs: int | None = None
    last_heartbeat: datetime
    registered_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}
