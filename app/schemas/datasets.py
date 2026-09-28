"""Wire shapes for training datasets."""
import re
import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

#: What a set is for (migration 103). See Dataset.kind.
DatasetKind = Literal["character", "composition", "regularization"]
#: The class words a regularization pool can stand in for. Not "person": no character is
#: registered under it, so a pool for it would be a pool nothing trains against.
RegClass = Literal["woman", "man"]

#: Same character class the image folders use, because a dataset's uploads land in one and an S3
#: prefix that needs escaping is a prefix nobody can type.
NAME_RE = re.compile(r"^[A-Za-z0-9 _.@-]+$")


class DatasetCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    tags: str | None = Field(default=None, max_length=500)
    notes: str | None = None
    #: Optional at creation: a set is often created before anyone decides whose it is.
    #: Validated the same way as on PATCH when given.
    kind: DatasetKind | None = None
    character: str | None = Field(default=None, max_length=64)
    reg_class: RegClass | None = None

    @field_validator("name")
    @classmethod
    def _safe(cls, v: str) -> str:
        return _safe_name(v)


def _safe_name(v: str) -> str:
    v = v.strip()
    if not NAME_RE.match(v):
        raise ValueError("letters, numbers, spaces, and . _ - @ only")
    return v


class DatasetClone(BaseModel):
    """POST /datasets/{id}/clone (#356). Only the name: everything else is the source's.

    Validated exactly as a new set's name is, because that is what a clone is.
    """
    name: str = Field(min_length=1, max_length=100)

    @field_validator("name")
    @classmethod
    def _safe(cls, v: str) -> str:
        return _safe_name(v)


class DatasetLock(BaseModel):
    """POST /datasets/{id}/lock (#358). The reason is optional and shown on the lock chip."""
    reason: str | None = Field(default=None, max_length=500)

    @field_validator("reason")
    @classmethod
    def _blank_is_none(cls, v: str | None) -> str | None:
        # "" from an empty confirmation box is no reason, not an empty pair of parentheses.
        return (v or "").strip() or None


class DatasetUpdate(BaseModel):
    #: Deliberately has no locked_at / locked_reason: a manual lock is one-way (#358), and an
    #: unknown field in a PATCH is ignored, so a PATCH cannot set or clear one.
    name: str | None = Field(default=None, max_length=100)
    tags: str | None = Field(default=None, max_length=500)
    notes: str | None = None
    #: Replaces the list wholesale. Used to reorder or remove; adding is done by uploading.
    images: list[str] | None = None
    #: The image every other one is scored against. Settable directly so the console can clear
    #: it, or set it without immediately paying for a scoring pass.
    anchor_uri: str | None = None
    #: Ownership (migration 103). Unlike the fields above, an explicit null CLEARS these --
    #: "unassign this set" has to be sayable, and "" is not a kind. The route tells an
    #: absent field from a null through model_fields_set.
    kind: DatasetKind | None = None
    character: str | None = Field(default=None, max_length=64)
    reg_class: RegClass | None = None


class DatasetCaptionsRun(BaseModel):
    """POST /datasets/{id}/captions. Fills only missing captions unless `overwrite`."""
    overwrite: bool = False


class DatasetCaptionEdit(BaseModel):
    """PATCH /datasets/{id}/captions: one image's body, by URI. Blank deletes it."""
    uri: str
    caption: str = Field(max_length=2000)


class DatasetCaptionStatus(BaseModel):
    """Progress of a dataset's captioning. `running` is this process's view of it."""
    total: int
    captioned: int
    running: bool
    error: str | None = None


class DatasetRegularize(BaseModel):
    """POST /datasets/{id}/regularize: how many text-to-video renders to queue."""
    count: int = Field(ge=1, le=300)


class DatasetRegularizeStatus(BaseModel):
    """Where a regularization pool's renders are, and what this poll collected.

    requested = done + failed + running. `done` renders have had their frame collected into
    the set; `running` is queued or rendering (`pending` is the queued part of it).
    """
    requested: int
    done: int
    failed: int
    running: int
    pending: int = 0
    collected_now: int = 0
    images: int = 0


class DatasetTrainedBy(BaseModel):
    """One training run that used a set (#356), read from the run's recorded provenance."""
    job_id: str
    character: str
    version: int
    status: str


class DatasetResponse(BaseModel):
    id: uuid.UUID
    name: str
    tags: str | None = None
    notes: str | None = None
    images: list[str]
    prefix: str | None = None
    anchor_uri: str | None = None
    kind: str | None = None
    character: str | None = None
    reg_class: str | None = None
    #: {uri: body}, without the trigger prefix (migration 103).
    captions: dict[str, str] = Field(default_factory=dict)
    #: {uri: cos} against the anchor as of the last scoring pass; null = no face found.
    scores: dict[str, float | None] = Field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None
    #: LOCKED ONCE IT HAS TRAINED (#356): some run that is not failed or cancelled used this
    #: set, so a LoRA came -- or is coming -- of exactly these images and captions. Not
    #: columns: derived from training_jobs on every read, so a run that fails unlocks the set
    #: with nothing to keep in step. The ORM row has neither attribute, so a response built
    #: straight from it reads unlocked; the routes fill both (datasets._respond).
    locked: bool = False
    #: Every run that locks it, oldest first -- what the console's lock chip names.
    trained_by: list[DatasetTrainedBy] = Field(default_factory=list)
    #: LOCKED BY HAND (#358): columns, unlike the above. `locked` is true when either is set.
    locked_at: datetime | None = None
    locked_reason: str | None = None

    @field_validator("captions", "scores", mode="before")
    @classmethod
    def _none_is_empty(cls, v):
        # A row created by the ORM before its server default is read back holds None.
        return v or {}

    @property
    def image_count(self) -> int:
        return len(self.images)

    model_config = {"from_attributes": True}


class DatasetScore(BaseModel):
    """One image's likeness to the anchor."""
    uri: str
    #: None means no face was detected, which is not a low score but an absent one. The console
    #: has to say "no face" rather than rendering it as the worst match in the set.
    cos: float | None = None
    is_anchor: bool = False


class DatasetScores(BaseModel):
    anchor_uri: str
    #: buffalo_l's same-person floor. A line on a chart, not a delete rule.
    cos_floor: float
    scores: list[DatasetScore]
