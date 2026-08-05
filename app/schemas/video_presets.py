from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel


class LoraSlot(BaseModel):
    lora_id: str
    high_weight: float
    low_weight: float


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
    prompt: Optional[str] = None
    notes: Optional[str] = None


class VideoPresetUpdate(BaseModel):
    archived: Optional[bool] = None
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
    prompt: Optional[str] = None
    notes: Optional[str] = None


class VideoPresetResponse(BaseModel):
    archived: bool = False
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
    prompt: Optional[str] = None
    notes: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}
