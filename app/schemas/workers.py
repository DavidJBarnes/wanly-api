import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel


class WorkerRegister(BaseModel):
    friendly_name: str
    hostname: str
    ip_address: str
    comfyui_running: bool = False


class WorkerHeartbeat(BaseModel):
    comfyui_running: bool
    gpu_stats: dict[str, Any] | None = None
    sd_scripts: dict[str, Any] | None = None
    a1111: dict[str, Any] | None = None


class WorkerRename(BaseModel):
    friendly_name: str


class WorkerStatusUpdate(BaseModel):
    status: str


class WorkerDrain(BaseModel):
    after_jobs: int | None = None


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
    drain_after_jobs: int | None = None
    last_heartbeat: datetime
    registered_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}
