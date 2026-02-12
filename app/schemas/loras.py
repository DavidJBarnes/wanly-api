from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel


class LoraCreate(BaseModel):
    name: str
    description: Optional[str] = None
    trigger_words: Optional[str] = None
    default_prompt: Optional[str] = None
    source_url: Optional[str] = None
    high_url: Optional[str] = None
    low_url: Optional[str] = None
    default_high_weight: float = 1.0
    default_low_weight: float = 1.0


class LoraUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    trigger_words: Optional[str] = None
    default_prompt: Optional[str] = None
    source_url: Optional[str] = None
    default_high_weight: Optional[float] = None
    default_low_weight: Optional[float] = None


class LoraResponse(BaseModel):
    id: UUID
    name: str
    description: Optional[str]
    trigger_words: Optional[str]
    default_prompt: Optional[str]
    source_url: Optional[str]
    preview_image: Optional[str]
    high_file: Optional[str]
    high_s3_uri: Optional[str]
    low_file: Optional[str]
    low_s3_uri: Optional[str]
    default_high_weight: float
    default_low_weight: float
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class LoraListItem(BaseModel):
    id: UUID
    name: str
    trigger_words: Optional[str]
    preview_image: Optional[str]
    high_file: Optional[str]
    low_file: Optional[str]
    default_high_weight: float
    default_low_weight: float
    default_prompt: Optional[str]

    model_config = {"from_attributes": True}
