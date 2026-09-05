from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class ImageTagsUpdate(BaseModel):
    tags: Optional[str] = Field(None, max_length=500, description="Comma-separated tags")


class ImageSceneRequest(BaseModel):
    """Describe this image now. Both "first description" and "re-roll" are this call.

    Style and instruction mirror CaptionRequest so a one-off "try it shorter" is possible
    without moving the global setting.
    """

    style: Optional[str] = None
    instruction: Optional[str] = Field(default=None, max_length=2000)


class ImageSceneResponse(BaseModel):
    path: str
    # None means never described. Distinct from "" which nothing writes -- a blank caption
    # is a captioner failure, not a description.
    scene_description: Optional[str] = None
    # HOW it was described. A caption written under "terse" and one under "rich" are
    # different artefacts and the row has to say which this is.
    scene_instruction: Optional[str] = None
    scene_described_at: Optional[datetime] = None
    # Length is the thing being judged: the description sits beside a ~100-word arc, and
    # whether it is 25 or 80 words changes the balance between scene and motion.
    words: int = 0
