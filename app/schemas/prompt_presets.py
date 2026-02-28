from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel


class LoraSlot(BaseModel):
    lora_id: str
    high_weight: float
    low_weight: float


class PromptPresetCreate(BaseModel):
    name: str
    prompt: str
    loras: Optional[list[LoraSlot]] = None


class PromptPresetUpdate(BaseModel):
    name: Optional[str] = None
    prompt: Optional[str] = None
    loras: Optional[list[LoraSlot]] = None


class PromptPresetResponse(BaseModel):
    id: UUID
    name: str
    prompt: str
    loras: Optional[list[LoraSlot]] = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}
