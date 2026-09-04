"""Schemas for LTX characters and recipes."""

import uuid
from datetime import datetime
from typing import Optional, List

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


class ContentLora(BaseModel):
    """One content LoRA in a pose's chain (console#410).

    Per-stage strengths because stage 1 generates at half size from noise and stage 2
    refines the 2x-upscaled latent. Both default to 0.6 — the value resolve() hardcoded
    before any of this was configurable — so adding a LoRA and touching nothing renders it
    at the strength the validated graph already applied.
    """

    name: str = Field(min_length=1, max_length=256)
    # Bounded at 2.0 to match the ENGINE's own bound. A wider bound here would accept a
    # value the console stores happily and the engine then rejects with a 422, ten minutes
    # into a claimed segment.
    s1: float = Field(default=0.6, ge=0, le=2)
    s2: float = Field(default=0.6, ge=0, le=2)


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
    # Motion/act LoRAs for this pose, IN APPLICATION ORDER. Empty or absent means none,
    # which is what every existing pose does.
    #
    # Capped at 4 to match LtxRequest.loras' own max_length. Four LoRAs on one chain is
    # already a lot of competition for the same weights as the character LoRA.
    content_loras: Optional[List[ContentLora]] = Field(default=None, max_length=4)
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
    # An empty list CLEARS them; None leaves them alone. Those are different intents and
    # exclude_none in the route keeps them distinguishable.
    content_loras: Optional[List[ContentLora]] = Field(default=None, max_length=4)
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
    content_loras: List[ContentLora] = Field(default_factory=list)
    checkpoint: Optional[str]
    validated: bool
    created_at: datetime
    updated_at: Optional[datetime]
