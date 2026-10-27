from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field

from app.schemas.app_settings import CaptionInstruction, CaptionStyle


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


class ImagesInUseRequest(BaseModel):
    """Which of these images is something still holding? (console#594)

    The bulk-delete pre-check: one call for the whole selection instead of a refused DELETE
    per image. Capped so a runaway selection cannot turn into an unbounded IN list.
    """

    paths: list[str] = Field(..., min_length=1, max_length=2000,
                             description="s3:// URIs in the images bucket")


class ImageSceneRequest(BaseModel):
    """Describe this image now. Both "first description" and "re-roll" are this call.

    Style and instruction mirror CaptionRequest so a one-off "try it shorter" is possible
    without moving the global setting.
    """

    style: Optional[str] = None
    instruction: Optional[CaptionInstruction] = None
    # Which halves to make (console#590). Only "scene" exists since wanly-api#481; omitted or
    # empty means the scene. "motion" is still ACCEPTED so an older console gets a clear
    # answer rather than a schema error: dropped beside "scene", a 422 on its own.
    halves: Optional[list[Literal["scene", "motion"]]] = None


class CaptionTryRequest(BaseModel):
    """Run the caption prompts from the Settings editors on one image, storing nothing.

    console#555. Each field has THREE states, and the console uses all of them:

        omitted / null   use what Settings has SAVED -- the "saved prompt" column
        ""               use the built-in DEFAULT -- what Reset would give
        text             use exactly this (unsaved) text -- the "current text" column

    so one request shape answers "what does the saved prompt say" and "what would my edit
    say" without the console restating any default. Field names match the settings keys the
    editors write, rather than ImageSceneRequest's, so the page sends the same values it
    would save.
    """

    caption_style: Optional[CaptionStyle] = None
    caption_instruction: Optional[CaptionInstruction] = None


class CaptionTryResponse(BaseModel):
    caption: str
    words: int = 0
    # The exact text the captioner was sent, so the editor can show what it actually read.
    caption_instruction_used: str


class CaptionRequester(BaseModel):
    """A held job a caption is being made for (console#590)."""
    job_id: str
    name: Optional[str] = None


class CaptionTicket(BaseModel):
    """The scene caption of one image, in the background (console#564; app/caption_tickets.py).

    status: "queued" (position 1 = next up), "running" (position 0), "done" or "failed".
    Null status means no caption of the image is in flight or remembered.
    """
    path: str
    ticket_id: Optional[str] = None
    status: Optional[str] = None
    position: Optional[int] = None
    #: Everything unfinished in the caption queue, including the one in progress.
    depth: int = 0
    #: "scene" (console#590): the half this ticket makes. The only half since wanly-api#481.
    half: Optional[str] = None
    #: The same as `half`. Was "pair" before the halves were split.
    mode: Optional[str] = None
    #: Who asked first: "describe" or "hold".
    origin: Optional[str] = None
    error: Optional[str] = None
    #: Failed because the box beside the captioner is rendering -- try again later.
    busy: bool = False
    #: The held jobs this caption is for -- "requested by job ..." (console#590).
    requested_by: list[CaptionRequester] = []
    #: True when this request joined a caption already in flight instead of queueing one.
    joined: bool = False
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    #: While unfinished, which caption lane it is in ("scene"). `position` and `depth` are
    #: that lane's.
    lane: Optional[str] = None
    #: POST /images/scene/describe: every half's ticket (just the scene's).
    tickets: Optional[list["CaptionTicket"]] = None
    #: GET /images/scene/status: the scene's ticket (in flight, else the last finished).
    scene: Optional["CaptionTicket"] = None


class CaptionQueueEntry(BaseModel):
    """One place in the caption queue."""
    path: str
    #: "describe", "hold" (both save words on the image), "dataset" or "try" (they do not).
    kind: str
    status: str  # "running" | "queued"
    #: Within its lane.
    position: int
    ticket_id: Optional[str] = None
    #: Which lane: "scene", the only one since wanly-api#481.
    lane: str = "scene"
    #: The held jobs a ticket is for (console#590); empty when a person asked.
    requested_by: list[CaptionRequester] = []


class CaptionLane(BaseModel):
    """One caption lane: its own captioner, its own line."""
    name: str
    depth: int = 0
    waiting: int = 0
    running: Optional[str] = None


class CaptionQueueStatus(BaseModel):
    """The captioner's queue, for a view that is not about one image.

    Asked for because the per-image position only exists inside the modal of an image you
    are already describing -- there was no answer to "how is the queue looking?" without
    opening one.

    Since console#564 it also lists every entry and every recently finished caption ticket,
    so ONE poll lets the console mark any image on any page -- "In caption queue (#3)",
    "Captioning…", "Failed: retry" -- without a request per image.
    """
    #: Everything unfinished, including the one in progress.
    depth: int = 0
    #: How many are still waiting to start.
    waiting: int = 0
    #: The image being captioned right now, so the view can name it rather than just count.
    running: Optional[str] = None
    #: The whole line, running first.
    entries: list[CaptionQueueEntry] = []
    #: The last finished ticket of each image that has one remembered (done or failed),
    #: newest first.
    recent: list[CaptionTicket] = []
    #: Each lane on its own (wanly-console#572) -- one, the scene's, since wanly-api#481.
    lanes: list[CaptionLane] = []
    #: The scene captioner: {url, model, up, why, fallback}. up=False means scene captions
    #: are going to the fallback (the image-description captioner) because the scene service
    #: is down.
    scene_captioner: Optional[dict] = None


class ImageSceneResponse(BaseModel):
    path: str
    # None means never described. Distinct from "" which nothing writes -- a blank caption
    # is a captioner failure, not a description.
    scene_description: Optional[str] = None
    # HOW it was described. A caption written under "terse" and one under "rich" are
    # different artefacts and the row has to say which this is.
    scene_instruction: Optional[str] = None
    scene_described_at: Optional[datetime] = None
    # The Sulphur-2 prompt enhancer's video prompt (wanly-api#480). None = never enhanced.
    enhanced_prompt: Optional[str] = None
    enhanced_prompt_at: Optional[datetime] = None
    # Length is the thing being judged: the description sits beside a ~100-word arc, and
    # whether it is 25 or 80 words changes the balance between scene and action.
    words: int = 0
    # THE QUEUE IN FRONT OF THE CAPTIONER (app/caption_queue.py). ollama is one slot, so
    # describes run strictly one at a time; these say where this image sits in that line so
    # the UI can show "3rd of 7" rather than a spinner indistinguishable from a hang.
    # All null/0 when the path is not queued, which is the normal idle case.
    queue_status: Optional[str] = None
    queue_position: Optional[int] = None
    queue_depth: int = 0
    # The image's caption ticket (console#564): the one in flight, else the last finished one
    # still remembered. Null when there is neither. `scene_caption` is the same ticket
    # (console#590 named it per half).
    caption: Optional[CaptionTicket] = None
    scene_caption: Optional[CaptionTicket] = None


class EnhancePromptResponse(BaseModel):
    """POST /images/enhance-prompt (wanly-api#480): the saved prompt and what it cost."""
    path: str
    enhanced_prompt: str
    enhanced_prompt_at: datetime
    words: int
    #: Milliseconds the enhancer took, and whether it had to load the model first.
    ms: Optional[int] = None
    cold: bool = False
