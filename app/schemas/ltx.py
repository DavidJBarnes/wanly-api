"""Schemas for LTX characters and recipes."""

import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class LtxCharacterCreate(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    char_lora: str = Field(min_length=1)
    # Fills every pose's <TRIGGER> placeholder. Defaults to the character's own name, which
    # is what all three seeded characters use.
    trigger: Optional[str] = Field(default=None, max_length=64)
    # Per-stage, never flat — stage 1 decides the body, stage 2 resolves the face.
    strength_stage_1: float = 0.8
    strength_stage_2: float = 1.5


class LtxCharacterResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    char_lora: str
    trigger: str
    strength_stage_1: float
    strength_stage_2: float


class LtxRecipeCreate(BaseModel):
    """A pose. Character-agnostic: the prompt carries <TRIGGER>, not a name."""

    name: str = Field(min_length=1, max_length=128)
    prompt_template: str = Field(min_length=1)
    # NULL means the global stack's negative, which is true of every seeded recipe.
    negative_prompt: Optional[str] = None
    frames: Optional[int] = Field(default=None, gt=0)
    validated: bool = False


class LtxRecipeUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=128)
    prompt_template: Optional[str] = Field(default=None, min_length=1)
    negative_prompt: Optional[str] = None
    frames: Optional[int] = Field(default=None, gt=0)
    validated: Optional[bool] = None


class LtxRecipeResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    prompt_template: str
    negative_prompt: Optional[str]
    frames: Optional[int]
    validated: bool
    created_at: datetime
    updated_at: Optional[datetime]
