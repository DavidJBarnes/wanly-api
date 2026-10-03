from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field

from app.schemas.app_settings import CaptionInstruction, CaptionStyle, MotionStyle, MotionTemplate


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
    without moving the global setting. The motion pair overrides the same way (#326).
    """

    style: Optional[str] = None
    instruction: Optional[CaptionInstruction] = None
    motion_style: Optional[str] = None
    # A motion TEMPLATE since console#555, validated like the saved setting.
    motion_instruction: Optional[MotionTemplate] = None
    # Which halves to make (console#590): ["scene"] (what tagging does, and Redo scene),
    # ["motion"] (Describe motion / Redo motion -- grounded on the saved scene) or both.
    # Omitted or empty: both, as before, for a client that predates the split.
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
    motion_style: Optional[MotionStyle] = None
    motion_template: Optional[MotionTemplate] = None


class CaptionTryResponse(BaseModel):
    caption: str
    words: int = 0
    # None when the motion half failed (motion_error says why) or is switched off
    # (motion_enabled false) -- the same partial-success shape as POST /images/scene.
    motion: Optional[str] = None
    motion_words: int = 0
    motion_error: Optional[str] = None
    motion_enabled: bool = True
    # The exact text each half was sent -- the rendered template, not the template -- so the
    # editor can show what the captioner actually read.
    caption_instruction_used: str
    motion_instruction_used: Optional[str] = None


class CaptionRequester(BaseModel):
    """A held job a caption is being made for (console#590)."""
    job_id: str
    name: Optional[str] = None


class CaptionTicket(BaseModel):
    """One HALF of the caption of one image, in the background (console#564, #590;
    app/caption_tickets.py).

    status: "queued" (position 1 = next up), "running" (position 0), "done" or "failed".
    Null status means no caption of the image is in flight or remembered.
    """
    path: str
    ticket_id: Optional[str] = None
    status: Optional[str] = None
    position: Optional[int] = None
    #: Everything unfinished in the caption queue, including the one in progress.
    depth: int = 0
    #: "scene" or "motion" (console#590): the half this ticket makes, and nothing else.
    half: Optional[str] = None
    #: The same as `half`. Was "pair"/"motion" before the halves were split.
    mode: Optional[str] = None
    #: Who asked first: "describe" or "hold".
    origin: Optional[str] = None
    error: Optional[str] = None
    #: Failed because the box beside the captioner is rendering -- try again later.
    busy: bool = False
    #: Always null since console#590 (a ticket is one half; a motion failure is the motion
    #: ticket's own `error`). Kept for clients that read it.
    motion_error: Optional[str] = None
    #: The held jobs this caption is for -- "Motion requested by job ..." (console#590).
    requested_by: list[CaptionRequester] = []
    #: True when this request joined a caption already in flight instead of queueing one.
    joined: bool = False
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    #: While unfinished, which caption lane it is in -- the half's own (console#572).
    #: `position` and `depth` are that lane's.
    lane: Optional[str] = None
    #: POST /images/scene/describe: every half's ticket, scene first.
    tickets: Optional[list["CaptionTicket"]] = None
    #: GET /images/scene/status: each half's ticket (in flight, else the last finished).
    scene: Optional["CaptionTicket"] = None
    motion: Optional["CaptionTicket"] = None


class CaptionQueueEntry(BaseModel):
    """One place in the caption queue."""
    path: str
    #: "describe", "hold" (both save words on the image), "dataset" or "try" (they do not).
    kind: str
    status: str  # "running" | "queued"
    #: Within its lane.
    position: int
    ticket_id: Optional[str] = None
    #: "scene" or "motion" (wanly-console#572). For a caption ticket, the half it makes.
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
    #: Each lane on its own (wanly-console#572). depth/waiting above count both; `running`
    #: above is the scene lane's, as it always was.
    lanes: list[CaptionLane] = []
    #: The scene captioner: {url, model, up, why, fallback}. up=False means scene captions
    #: are going to the fallback (the motion captioner) because the scene service is down.
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
    # The image's caption ticket (console#564): the one in flight, else the last finished one
    # still remembered. Null when there is neither. Per half since console#590:
    # scene_caption / motion_caption; `caption` is whichever is in flight (scene first), else
    # the newer finish.
    caption: Optional[CaptionTicket] = None
    scene_caption: Optional[CaptionTicket] = None
    motion_caption: Optional[CaptionTicket] = None
