from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel

from app.schemas.prompt_presets import LoraSlot


class VideoPresetCreate(BaseModel):
    name: str
    lightx2v_strength_high: Optional[float] = None
    lightx2v_strength_low: Optional[float] = None
    cfg_high: Optional[float] = None
    cfg_low: Optional[float] = None
    steps_total: Optional[int] = None
    high_noise_steps: Optional[int] = None
    flow_shift: Optional[float] = None
    sampler_name: Optional[str] = None
    scheduler: Optional[str] = None
    loras: Optional[list[LoraSlot]] = None


class VideoPresetUpdate(BaseModel):
    name: Optional[str] = None
    lightx2v_strength_high: Optional[float] = None
    lightx2v_strength_low: Optional[float] = None
    cfg_high: Optional[float] = None
    cfg_low: Optional[float] = None
    steps_total: Optional[int] = None
    high_noise_steps: Optional[int] = None
    flow_shift: Optional[float] = None
    sampler_name: Optional[str] = None
    scheduler: Optional[str] = None
    loras: Optional[list[LoraSlot]] = None


class VideoPresetResponse(BaseModel):
    id: UUID
    name: str
    lightx2v_strength_high: Optional[float] = None
    lightx2v_strength_low: Optional[float] = None
    cfg_high: Optional[float] = None
    cfg_low: Optional[float] = None
    steps_total: Optional[int] = None
    high_noise_steps: Optional[int] = None
    flow_shift: Optional[float] = None
    sampler_name: Optional[str] = None
    scheduler: Optional[str] = None
    loras: Optional[list[LoraSlot]] = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}
