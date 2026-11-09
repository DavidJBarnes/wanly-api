"""Character face loops (wanly-api#495, wanly-console#742)."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.face_loop_presets import PRESETS


class FaceLoopCreate(BaseModel):
    preset: str = "calm"
    start_image: str = Field(min_length=6, description="s3:// URI of a close-up, front-facing photo")
    label: Optional[str] = Field(None, max_length=120)
    motion: Optional[str] = Field(None, max_length=2000, description="custom preset only: what the face does")

    @model_validator(mode="after")
    def _check(self):
        if self.preset not in PRESETS:
            raise ValueError(f"unknown preset {self.preset!r}; one of {sorted(PRESETS)}")
        if self.preset == "custom" and not (self.motion or "").strip():
            raise ValueError("a custom face loop needs `motion`: what the face does")
        if not self.start_image.startswith("s3://"):
            raise ValueError("start_image must be an s3:// URI")
        return self


class FaceLoopUpdate(BaseModel):
    label: str = Field(min_length=1, max_length=120)


class FaceLoopResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    character_id: uuid.UUID
    character: Optional[str] = None
    label: str
    preset: str
    prompt: str
    start_image: str
    job_id: Optional[uuid.UUID] = None
    status: str
    # The render's own state while status is "rendering" (pending / processing / failed ...).
    render_status: Optional[str] = None
    asset_prefix: Optional[str] = None
    window_start: Optional[float] = None
    window_end: Optional[float] = None
    stats: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None


class FaceLoopPreset(BaseModel):
    id: str
    label: str
    motion: str


class FaceLoopPhoto(BaseModel):
    """A candidate start photo and how well it suits a face loop (bigger, frontal, mouth closed,
    eyes open, sharp = better). `suitable` is the hard check; `score` orders them."""
    uri: str
    score: float
    suitable: bool
    face_px: int
    yaw: float
    pitch: float
    mouth_closed: bool
    eyes_open: bool
