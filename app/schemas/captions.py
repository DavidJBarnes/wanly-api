from typing import Optional

from pydantic import BaseModel, Field


class CaptionRequest(BaseModel):
    # An s3:// URI. Deliberately not a raw upload: the image is already in S3 by the time
    # anyone is composing a segment, and accepting bytes here would make this an upload
    # endpoint with a different security surface.
    image_uri: str = Field(min_length=1)
    # Override the saved setting for this one call, so the console can offer "try it
    # shorter" without changing the global default.
    style: Optional[str] = None
    instruction: Optional[str] = Field(default=None, max_length=2000)


class CaptionResponse(BaseModel):
    caption: str
    # Which instruction produced this. Returned so the caller can record HOW the frame was
    # described, not only what was said.
    instruction: str
    # Surfaced because length is the thing being judged: the caption sits beside a ~100-word
    # arc, and whether it is 25 or 80 words changes the balance between scene and motion.
    words: int
