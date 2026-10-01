"""Schemas for LTX characters and recipes."""

import uuid
from datetime import datetime
from typing import Annotated, Literal, Optional, List

from pydantic import BaseModel, ConfigDict, Field

#: The word a LoRA's caption bound its trigger to (wanly-console#487). The same three the
#: training request takes, because this is what that request's caption recorded.
Gender = Literal["woman", "man", "person"]
#: Which identity reference renders (migration 107): the 1536x1024 character sheet, or the
#: face close-up. Must name one the character actually has.
IdentityMode = Literal["sheet", "face"]
#: An Image Repo reference is an S3 URI. A presigned https URL would expire inside the row.
S3Uri = Annotated[str, Field(min_length=6, max_length=1024, pattern=r"^s3://[^/]+/.+")]


class LtxCharacterCreate(BaseModel):
    """Register a character: the registry is where a trigger and a gender come from (#352).

    `char_lora` is OPTIONAL since #352 and defaults to "none", the value that already means
    "render on the base model" everywhere a character is read (recipe_blob, the claim's
    model requirements, the daemon). A character is registered BEFORE it trains -- its
    trigger and gender are what the training route captions with -- so there is no LoRA
    to name yet, and "none" is the honest one.

    A PAIR (`kind="pair"`) names its two solo `members`; its trigger is derived from theirs
    as the joined phrase and its gender is None (the phrase carries both). Anything sent
    for either is ignored rather than trusted: the phrase must be exactly what the captions
    will teach.

    A LoRA, A SHEET, OR BOTH (wanly-console#581). `sheet_uri` / `face_ref_uri` are Image Repo
    images the engine conditions on (wanly-gpu-docker#156). A character sent with a reference
    and no LoRA is SHEET-ONLY: its char_lora is stored NULL rather than "none", it needs no
    trigger or strengths, and <TRIGGER> fills from `description`. `identity_mode` defaults to
    the sheet when there is one, else the face.
    """
    name: str = Field(min_length=1, max_length=64)
    #: Omitted or null: "none". Stored as "none" rather than NULL because every reader of
    #: a character already treats that string as "no LoRA" (see the class docstring).
    char_lora: Optional[str] = Field(default=None, min_length=1)
    # Fills every pose's <TRIGGER> placeholder. Defaults to the character's own name, which
    # is what all three seeded characters use.
    trigger: Optional[str] = Field(default=None, max_length=64)
    # Renders beside the trigger, "p@yton, woman", exactly as the LoRA's caption read.
    gender: Optional[Gender] = None
    # Per-stage, never flat — stage 1 decides the body, stage 2 resolves the face.
    strength_stage_1: float = 0.8
    strength_stage_2: float = 1.5
    image_uri: Optional[str] = None
    kind: Literal["solo", "pair"] = "solo"
    members: Optional[List[str]] = Field(default=None, max_length=2)
    sheet_uri: Optional[S3Uri] = None
    face_ref_uri: Optional[S3Uri] = None
    identity_mode: Optional[IdentityMode] = None
    description: Optional[str] = Field(default=None, max_length=255)


class LtxCharacterResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    char_lora: Optional[str] = None
    #: None for a sheet-only character (migration 107).
    trigger: Optional[str] = None
    gender: Optional[Gender] = None
    strength_stage_1: float
    strength_stage_2: float
    image_uri: Optional[str] = None
    #: solo | pair (migration 103). A row predating it reads solo.
    kind: str = "solo"
    members: Optional[List[str]] = None
    base_checkpoint: Optional[str] = None
    trained_from: Optional[list] = None
    #: Preselected by the console's modals (wanly-console#543). At most one is true.
    is_default: bool = False
    #: The identity reference (migration 107): Image Repo URIs, which one renders, and the
    #: words <TRIGGER> fills with when there is no trigger.
    sheet_uri: Optional[str] = None
    face_ref_uri: Optional[str] = None
    identity_mode: Optional[str] = None
    description: Optional[str] = None


class LtxCharacterUpdate(BaseModel):
    """Every field optional: a PATCH that only moves a strength must not restate the LoRA.

    `trigger` does NOT re-default to the name here, unlike create. On create an absent
    trigger means "no opinion", and the name is the best guess. On update an absent trigger
    means "leave it alone", and quietly rewriting it to the new name would silently change
    every pose's rendered prompt as a side effect of a rename.

    Trigger, gender, kind and members are LOCKED once the character has trained (#352) --
    see update_character for the one way to change them anyway.
    """

    name: Optional[str] = Field(default=None, min_length=1, max_length=64)
    char_lora: Optional[str] = Field(default=None, min_length=1)
    trigger: Optional[str] = Field(default=None, min_length=1, max_length=255)
    # Sent as null, it clears: a character whose LoRA trained on a bare caption should not
    # keep rendering a gender it never bound.
    gender: Optional[Gender] = None
    strength_stage_1: Optional[float] = Field(default=None, ge=0)
    strength_stage_2: Optional[float] = Field(default=None, ge=0)
    image_uri: Optional[str] = None
    kind: Optional[Literal["solo", "pair"]] = None
    members: Optional[List[str]] = Field(default=None, max_length=2)
    # The identity reference (migration 107). Sent as null, each clears. `char_lora: null`
    # removes the LoRA outright -- allowed only while a reference remains (a character must
    # be one or the other); "none" is still the way to detach and re-register (#352).
    sheet_uri: Optional[S3Uri] = None
    face_ref_uri: Optional[S3Uri] = None
    identity_mode: Optional[IdentityMode] = None
    description: Optional[str] = Field(default=None, max_length=255)


class LtxBookCreate(BaseModel):
    """A named shelf of poses."""

    name: str = Field(min_length=1, max_length=64)
    description: Optional[str] = None


class LtxBookUpdate(BaseModel):
    """Every field optional: a rename must not restate the description, and vice versa."""

    name: Optional[str] = Field(default=None, min_length=1, max_length=64)
    description: Optional[str] = None


class LtxBookResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    description: Optional[str]
    #: How many poses are filed here. Not a column — assembled by the route so the console
    #: can warn before a delete rather than discovering the 409.
    recipe_count: int = 0
    created_at: datetime


class ContentLora(BaseModel):
    """One content LoRA in a pose's chain (console#410).

    Per-stage strengths because stage 1 generates at half size from noise and stage 2
    refines the 2x-upscaled latent. Both default to 0.6 — the value resolve() hardcoded
    before any of this was configurable — so adding a LoRA and touching nothing renders it
    at the strength the validated graph already applied.
    """

    name: str = Field(min_length=1, max_length=256)
    # Bounded at 2.0 to match the ENGINE's own bound. A wider bound here would accept a
    # value the console stores happily and the engine then rejects with a 422, ten minutes
    # into a claimed segment.
    s1: float = Field(default=0.6, ge=0, le=2)
    s2: float = Field(default=0.6, ge=0, le=2)


class LtxRecipeCreate(BaseModel):
    """A pose. Character-agnostic: the prompt carries <TRIGGER>, not a name."""

    name: str = Field(min_length=1, max_length=128)
    prompt_template: str = Field(min_length=1)
    # NULL means the global stack's negative, which is true of every seeded recipe.
    negative_prompt: Optional[str] = None
    frames: Optional[int] = Field(default=None, gt=0)
    # A video CRF for the conditioning frame. NULL uses the stack's value. 0 is meaningful:
    # it bypasses the encode. Bounded at 51 -- the node accepts 100, but x264's scale ends
    # at 51 and anything above it is nominal.
    img_compression: Optional[int] = Field(default=None, ge=0, le=51)
    # Motion/act LoRAs for this pose, IN APPLICATION ORDER. Empty or absent means none,
    # which is what every existing pose does.
    #
    # Capped at 4 to match LtxRequest.loras' own max_length. Four LoRAs on one chain is
    # already a lot of competition for the same weights as the character LoRA.
    content_loras: Optional[List[ContentLora]] = Field(default=None, max_length=4)
    # Base model for this pose. NULL uses the stack's. A filename as ComfyUI lists it;
    # the engine appends .safetensors when missing.
    checkpoint: Optional[str] = Field(default=None, max_length=256)
    # The book this pose is filed under. Absent means "the default book" — the route resolves
    # it, so creating a pose can never 400 or 409 on a field the caller did not think about.
    book_id: Optional[uuid.UUID] = None


class LtxRecipeUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=128)
    prompt_template: Optional[str] = Field(default=None, min_length=1)
    negative_prompt: Optional[str] = None
    frames: Optional[int] = Field(default=None, gt=0)
    img_compression: Optional[int] = Field(default=None, ge=0, le=51)
    # An empty list CLEARS them; None leaves them alone. Those are different intents and
    # exclude_none in the route keeps them distinguishable.
    content_loras: Optional[List[ContentLora]] = Field(default=None, max_length=4)
    # Base model for this pose. NULL uses the stack's. A filename as ComfyUI lists it;
    # the engine appends .safetensors when missing.
    checkpoint: Optional[str] = Field(default=None, max_length=256)
    # Moving a pose between books. None leaves it where it is.
    book_id: Optional[uuid.UUID] = None


class LtxRecipeResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    prompt_template: str
    negative_prompt: Optional[str]
    frames: Optional[int]
    img_compression: Optional[int]
    content_loras: List[ContentLora] = Field(default_factory=list)
    checkpoint: Optional[str]
    book_id: uuid.UUID
    book_name: Optional[str] = None
    #: Preselected by the console's modals (wanly-console#543). At most one is true.
    is_default: bool = False
    created_at: datetime
    updated_at: Optional[datetime]
