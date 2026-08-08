from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class ReservationCreate(BaseModel):
    name: str
    # The window. Bounded because an unbounded reservation is just a pod launch you cannot see
    # coming.
    minutes: int = Field(default=30, ge=1, le=720)
    # Optional, and the reason it exists here rather than at launch: a reservation can fire
    # unattended, so a drain policy is what turns "get me a 4090" into a bounded instruction.
    drain_after_jobs: int | None = Field(default=None, ge=1, le=100)
    # None means the server default. Validated against the same allowlist the launcher uses.
    gpu_type_id: str | None = None


class ReservationResponse(BaseModel):
    id: UUID
    name: str
    status: str
    expires_at: datetime
    drain_after_jobs: int | None = None
    gpu_type_id: str | None = None
    pod_id: str | None = None
    error: str | None = None
    attempts: int
    created_at: datetime

    model_config = {"from_attributes": True}
