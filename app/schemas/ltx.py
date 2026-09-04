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


class LtxCharacterUpdate(BaseModel):
    """Every field optional: a PATCH that only moves a strength must not restate the LoRA.

    `trigger` does NOT re-default to the name here, unlike create. On create an absent
    trigger means "no opinion", and the name is the best guess. On update an absent trigger
    means "leave it alone", and quietly rewriting it to the new name would silently change
    every pose's rendered prompt as a side effect of a rename.
    """

    name: Optional[str] = Field(default=None, min_length=1, max_length=64)
    char_lora: Optional[str] = Field(default=None, min_length=1)
    trigger: Optional[str] = Field(default=None, min_length=1, max_length=64)
    strength_stage_1: Optional[float] = Field(default=None, ge=0)
    strength_stage_2: Optional[float] = Field(default=None, ge=0)


class LtxRecipeCreate(BaseModel):
    """A pose. Character-agnostic: the prompt carries <TRIGGER>, not a name."""

    name: str = Field(min_length=1, max_length=128)
    prompt_template: str = Field(min_length=1)
    # NULL means the global stack's negative, which is true of every seeded recipe.
    negative_prompt: Optional[str] = None
    frames: Optional[int] = Field(default=None, gt=0)
    # A video CRF for the conditioning frame. NULL uses the stack's value. 0 is meaningful:
    # it bypasses the encode. Bounded at 51 -- the node accepts 100, but x264's scale ends
    # at 51 and anything above it is nominal.
    img_compression: Optional[int] = Field(default=None, ge=0, le=51)
    # The motion/act LoRA for this pose. NULL uses the stack's value, which is "none" — so a
    # pose that says nothing gets no content LoRA, which is the behaviour every existing
    # pose has today. "none" is also accepted explicitly, to mean "deliberately off".
    content_lora: Optional[str] = Field(default=None, max_length=256)
    # Per-stage, like the character strengths. Bounded at 2.0 to match the ENGINE's own
    # bound on these fields (engine/app.py Lora and content_s1/s2). A wider bound here would
    # accept a value the console could store and then fail at render time with a 422, ten
    # minutes into a claimed segment -- the validation has to be the tighter of the two.
    content_s1: Optional[float] = Field(default=None, ge=0, le=2)
    content_s2: Optional[float] = Field(default=None, ge=0, le=2)
    # Base model for this pose. NULL uses the stack's. A filename as ComfyUI lists it;
    # the engine appends .safetensors when missing.
    checkpoint: Optional[str] = Field(default=None, max_length=256)
    validated: bool = False


class LtxRecipeUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=128)
    prompt_template: Optional[str] = Field(default=None, min_length=1)
    negative_prompt: Optional[str] = None
    frames: Optional[int] = Field(default=None, gt=0)
    img_compression: Optional[int] = Field(default=None, ge=0, le=51)
    content_lora: Optional[str] = Field(default=None, max_length=256)
    content_s1: Optional[float] = Field(default=None, ge=0, le=2)
    content_s2: Optional[float] = Field(default=None, ge=0, le=2)
    # Base model for this pose. NULL uses the stack's. A filename as ComfyUI lists it;
    # the engine appends .safetensors when missing.
    checkpoint: Optional[str] = Field(default=None, max_length=256)
    validated: Optional[bool] = None


class LtxRecipeResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    prompt_template: str
    negative_prompt: Optional[str]
    frames: Optional[int]
    img_compression: Optional[int]
    content_lora: Optional[str]
    content_s1: Optional[float]
    content_s2: Optional[float]
    checkpoint: Optional[str]
    validated: bool
    created_at: datetime
    updated_at: Optional[datetime]
