from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class ImageTagsUpdate(BaseModel):
    tags: Optional[str] = Field(None, max_length=500, description="Comma-separated tags")


class BulkImageTagsUpdate(BaseModel):
    """Apply the same tag set to many images at once (console#517).

    mode "add" merges the tags into each image's existing string; mode "remove" drops every
    whole-tag match. Both dedupe through tag_filter.normalise_tag on the server, which the
    per-image PATCH cannot do — it replaces the whole blob from what the browser last saw.
    """

    paths: list[str] = Field(..., min_length=1, description="s3:// URIs in the images bucket")
    tags: str = Field(..., min_length=1, max_length=500, description="Comma-separated tags")
    mode: str = Field("add", pattern="^(add|remove)$")


class ImageSceneRequest(BaseModel):
    """Describe this image now. Both "first description" and "re-roll" are this call.

    Style and instruction mirror CaptionRequest so a one-off "try it shorter" is possible
    without moving the global setting. The motion pair overrides the same way (#326).
    """

    style: Optional[str] = None
    instruction: Optional[str] = Field(default=None, max_length=2000)
    motion_style: Optional[str] = None
    motion_instruction: Optional[str] = Field(default=None, max_length=2000)


class CaptionQueueStatus(BaseModel):
    """The captioner's queue, for a view that is not about one image.

    Asked for because the per-image position only exists inside the modal of an image you
    are already describing -- there was no answer to "how is the queue looking?" without
    opening one.
    """
    #: Everything unfinished, including the one in progress.
    depth: int = 0
    #: How many are still waiting to start.
    waiting: int = 0
    #: The image being captioned right now, so the view can name it rather than just count.
    running: Optional[str] = None


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
    # The motion half (#326): the frame read as the first frame of a 10-second clip.
    # None means "no motion caption" — either described before #326, or the motion call
    # failed while the static half succeeded. Both render as an absent section; a re-roll
    # regenerates both.
    motion_description: Optional[str] = None
    motion_instruction: Optional[str] = None
    motion_described_at: Optional[datetime] = None
    motion_words: int = 0
    # Set when the static half succeeded and the motion half failed. The console shows it
    # as a warning rather than letting a missing paragraph look intentional.
    motion_error: Optional[str] = None
    # THE QUEUE IN FRONT OF THE CAPTIONER (app/caption_queue.py). ollama is one slot, so
    # describes run strictly one at a time; these say where this image sits in that line so
    # the UI can show "3rd of 7" rather than a spinner indistinguishable from a hang.
    # All null/0 when the path is not queued, which is the normal idle case.
    queue_status: Optional[str] = None
    queue_position: Optional[int] = None
    queue_depth: int = 0
