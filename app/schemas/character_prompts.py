"""A character's list of good working prompts (wanly-console#675)."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CharacterPromptBody(BaseModel):
    text: str = Field(max_length=20_000)

    @field_validator("text")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("a prompt cannot be blank")
        return v


class CharacterPromptResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    character_id: uuid.UUID
    text: str
    created_at: datetime
    updated_at: Optional[datetime] = None
