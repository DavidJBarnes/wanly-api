from typing import Optional

from pydantic import BaseModel, Field


class ImageTagsUpdate(BaseModel):
    tags: Optional[str] = Field(None, max_length=500, description="Comma-separated tags")
