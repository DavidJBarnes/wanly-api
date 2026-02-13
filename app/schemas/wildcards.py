from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel


class WildcardCreate(BaseModel):
    name: str
    options: list[str] = []


class WildcardUpdate(BaseModel):
    name: Optional[str] = None
    options: Optional[list[str]] = None


class WildcardResponse(BaseModel):
    id: UUID
    name: str
    options: list[str]
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}
