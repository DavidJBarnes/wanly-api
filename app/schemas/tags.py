from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class TitleTagCreate(BaseModel):
    name: str
    group: int


class TitleTagResponse(BaseModel):
    id: UUID
    name: str
    group: int
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}
