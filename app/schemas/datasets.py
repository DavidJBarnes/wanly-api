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
    #: Deliberately has no locked_at / locked_reason / unlocked_at: only /lock and /unlock
    #: (#358, #363) move them, and an unknown field in a PATCH is ignored, so a PATCH cannot
    #: set or clear one.
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
    """One training run that used a set, or an image: its character, version and arch."""
    job_id: str
    character: str
    version: int
    status: str
    #: ltx | sdxl -- "v1" is per arch (#402), so a badge needs both.
    arch: str | None = None
    created_at: datetime | None = None


class DatasetFaceSize(BaseModel):
    """One still's face measurement (#432). `face_px` is the LARGEST face's height in pixels at
    TRAINING size -- after the trainer's downscale to the 1024^2 area, which never upscales --
    and null when no face was found. Pose is insightface's, in degrees."""
    width: int | None = None
    height: int | None = None
    face_px: float | None = None
    face_h: float | None = None
    yaw: float | None = None
    pitch: float | None = None
    roll: float | None = None
    det_score: float | None = None
    #: How many faces the detector found. >1 means face_px may be the wrong person's.
    faces: int = 0
    #: The smaller of the two largest faces' heights at training size (#436): what a
    #: COMPOSITION set's photo is judged small by. Null with fewer than two faces; absent on an
    #: entry measured before #436 (the API re-measures those on a composition set).
    pair_px: float | None = None
    #: The two largest faces' boxes, [x1, y1, x2, y2] in source pixels, largest first (#436).
    boxes: list[list[float]] | None = None
    #: The crop "Fix small faces" made of this photo, if any: head-and-shoulders, or on a
    #: composition set two-person.
    crop_uri: str | None = None
    #: On an upscaled copy: the original it replaced in the set (still in S3).
    upscaled_from: str | None = None


class DatasetFixStatus(BaseModel):
    """GET/POST /datasets/{id}/fix-small-faces: the run's progress, and the set's tally."""
    running: bool = False
    #: waiting for another fix | measuring | upscaling | cropping | measuring results | done
    stage: str | None = None
    done: int = 0
    total: int = 0
    error: str | None = None
    #: The note the finished run appended to the set.
    summary: str | None = None


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
    #: {uri: face measurement} for stills (#432). An absent still has not been measured yet.
    faces: dict[str, DatasetFaceSize] = Field(default_factory=dict)
    #: Lineage (wanly-api#445): {derived_uri: {"from", "how", "at"}}.
    derived: dict[str, dict] = {}
    created_at: datetime | None = None
    updated_at: datetime | None = None
    #: READ-ONLY: locked by hand (#358) or archived (#419). Training no longer locks a set
    #: (#420) -- each run records what it trained on (GET /training/{id}/trained-on), so the
    #: set stays the subject's living set. Filled by the routes (datasets._respond).
    locked: bool = False
    #: Every run with a LoRA that trained on this set, oldest first. Information, not a lock.
    trained_by: list[DatasetTrainedBy] = Field(default_factory=list)
    #: {uri: the runs that trained on that image} -- the "used in v1, v3" badges (#422).
    #: Images no run used are absent. Filled by the routes.
    used_in: dict[str, list[DatasetTrainedBy]] = Field(default_factory=dict)
    #: LOCKED BY HAND (#358): optional, off by default.
    locked_at: datetime | None = None
    locked_reason: str | None = None
    #: When POST /datasets/{id}/unlock was last called (#363).
    unlocked_at: datetime | None = None
    #: ARCHIVED (#419): a version set folded into its subject's living set. Hidden, read-only.
    archived_at: datetime | None = None

    @field_validator("captions", "scores", "faces", mode="before")
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


class DatasetRunGroup(BaseModel):
    kind: str
    dataset_name: str | None = None
    #: This group trained on the dataset being asked about.
    here: bool = False


class DatasetRunPair(BaseModel):
    character: str
    #: The pair's composition set: the run's home. None when it is gone.
    dataset_id: str | None = None


class DatasetRun(BaseModel):
    """GET /datasets/{id}/runs (wanly-console#647): a run on a dataset's page."""
    job_id: str
    role: Literal["home", "pair_member", "orphan"]
    pair: DatasetRunPair | None = None
    groups: list[DatasetRunGroup] = []
