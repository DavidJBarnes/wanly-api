from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class TitleTagCreate(BaseModel):
    name: str
    group: int
    # Action tags only (wanly-api#486).
    description: str | None = None


class TitleTagUpdate(BaseModel):
    """PATCH /tags/{id}: an Action tag's description. Blank clears it."""
    description: str | None = None


class TitleTagResponse(BaseModel):
    id: UUID
    name: str
    group: int
    description: str | None = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}
