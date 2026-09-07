"""Wire shapes for training datasets."""
import re
import uuid
from datetime import datetime

from pydantic import BaseModel, Field, field_validator

#: Same character class the image folders use, because a dataset's uploads land in one and an S3
#: prefix that needs escaping is a prefix nobody can type.
NAME_RE = re.compile(r"^[A-Za-z0-9 _.@-]+$")


class DatasetCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    tags: str | None = Field(default=None, max_length=500)
    notes: str | None = None

    @field_validator("name")
    @classmethod
    def _safe(cls, v: str) -> str:
        v = v.strip()
        if not NAME_RE.match(v):
            raise ValueError("letters, numbers, spaces, and . _ - @ only")
        return v


class DatasetUpdate(BaseModel):
    name: str | None = Field(default=None, max_length=100)
    tags: str | None = Field(default=None, max_length=500)
    notes: str | None = None
    #: Replaces the list wholesale. Used to reorder or remove; adding is done by uploading.
    images: list[str] | None = None


class DatasetResponse(BaseModel):
    id: uuid.UUID
    name: str
    tags: str | None = None
    notes: str | None = None
    images: list[str]
    prefix: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @property
    def image_count(self) -> int:
        return len(self.images)

    model_config = {"from_attributes": True}
