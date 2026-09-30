"""The Image Edit tool's wire shapes (wanly-console#547).

`mode` is a field from day one, though phase 1 has one value: phase 2 (#548) adds "full" --
Qwen-Image-Edit on the 3090, with an instruction and a seed, run as an asynchronous job -- on
the same route, and a request that already says which engine it wants is what lets the two
share one dialog and one endpoint without guessing.
"""
import uuid
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.face_edit import MAX_PROMPT


class FaceExpression(BaseModel):
    """LivePortrait ExpressionEditor's parameters, all optional: a value laid over a preset.

    Ranges are the node's own. extra="forbid" so a misspelt axis is a 422 rather than an edit
    that silently ignored the one thing it was asked to do.
    """
    model_config = ConfigDict(extra="forbid")

    rotate_pitch: Optional[float] = Field(None, ge=-20, le=20)
    rotate_yaw: Optional[float] = Field(None, ge=-20, le=20)
    rotate_roll: Optional[float] = Field(None, ge=-20, le=20)
    blink: Optional[float] = Field(None, ge=-20, le=5)
    eyebrow: Optional[float] = Field(None, ge=-10, le=15)
    wink: Optional[float] = Field(None, ge=0, le=25)
    pupil_x: Optional[float] = Field(None, ge=-15, le=15)
    pupil_y: Optional[float] = Field(None, ge=-15, le=15)
    aaa: Optional[float] = Field(None, ge=-30, le=120)
    eee: Optional[float] = Field(None, ge=-20, le=15)
    woo: Optional[float] = Field(None, ge=-20, le=15)
    smile: Optional[float] = Field(None, ge=-0.3, le=1.3)


def _check_box(v: Optional[list[float]]) -> Optional[list[float]]:
    if v is not None and not (v[2] > v[0] and v[3] > v[1] and min(v) >= 0):
        raise ValueError("face_box must be [x1, y1, x2, y2] with x2 > x1, y2 > y1, all >= 0")
    return v


class _FaceChoice(BaseModel):
    """Which face, when the image has more than one (console#553). A box from
    POST /images/edit/faces, in the source's pixels -- preferred, because it names the face by
    where it is -- or an index into that call's left-to-right list. Neither: the face the node
    picks itself (the one nearest the horizontal centre), exactly as before."""
    face_index: Optional[int] = Field(None, ge=0, le=100)
    face_box: Optional[list[float]] = Field(None, min_length=4, max_length=4)

    @field_validator("face_box")
    @classmethod
    def _a_real_box(cls, v):
        return _check_box(v)


class HeadAngle(BaseModel):
    """Degrees, in the IMAGE's directions (app/full_edit.py HEAD_ANGLES): yaw < 0 turns the face
    toward the left edge of the picture, pitch > 0 raises the chin."""
    yaw: float = Field(0.0, ge=-90, le=90)
    pitch: float = Field(0.0, ge=-45, le=45)


class ImageEditRequest(_FaceChoice):
    #: The image to edit. Never overwritten: the result is always a new object.
    source_uri: str = Field(..., min_length=1, max_length=1000)
    #: "face": LivePortrait, answered inline with the saved image. "full" (#548): Qwen-Image-Edit
    #: on the 3090, answered at once with a JOB (202) -- poll GET /images/edit/jobs/{id}, then
    #: save the result with POST /images/edit/jobs/{id}/save.
    mode: Literal["face", "full"] = "face"
    # --- full mode: any mix of an instruction, a head angle (a named one or yaw/pitch) and an
    # expression preset (`preset`, one of app/full_edit.py EXPRESSIONS) -- #569 ---
    instruction: Optional[str] = Field(None, max_length=2000)
    angle: Optional[HeadAngle] = None
    head_preset: Optional[str] = Field(None, max_length=50)
    seed: Optional[int] = Field(None, ge=0, le=2**48)
    #: < 1 starts from the source and only partly redraws it; a head turn needs 1.0.
    denoise: Optional[float] = Field(None, gt=0, le=1)
    #: Named server-side (GET /images/edit/presets). Either or both; explicit values win.
    preset: Optional[str] = Field(None, max_length=50)
    expression: Optional[FaceExpression] = None
    #: "Describe the change" (console#550): read by the service against its keyword lexicon
    #: ("big smile, eyes closed, look left"). Instead of numbers, or with them -- in which case
    #: the numbers win, the service's rule. At least one of the three is required.
    prompt: Optional[str] = Field(None, max_length=MAX_PROMPT)
    #: Save into this dataset (appended to its image list) instead of the Image Repo.
    #: Refused with 409 when the set is locked (#356/#358), before anything runs.
    dataset_id: Optional[uuid.UUID] = None


class ImageEditPreviewRequest(_FaceChoice):
    source_uri: str = Field(..., min_length=1, max_length=1000)
    mode: Literal["face"] = "face"
    preset: Optional[str] = Field(None, max_length=50)
    expression: Optional[FaceExpression] = None
    prompt: Optional[str] = Field(None, max_length=MAX_PROMPT)


class ImageEditPreview(BaseModel):
    #: data: URI -- a capped-size JPEG, for the dialog's "after" pane only. Never stored.
    image: str
    params: dict[str, float]
    #: All twelve axes as the service applied them, zeros included -- what the editor sets its
    #: sliders to after a described change, so the user can fine-tune from there (#550).
    expression: dict[str, float]
    #: The service's account of where the numbers came from: "explicit", or
    #: "prompt:<lexicon hit>,…" -- and those hits as words, for the editor's chips.
    source: Optional[str] = None
    matched_terms: list[str] = []
    width: int
    height: int
    device: Optional[str] = None
    device_reason: Optional[str] = None
    elapsed_ms: Optional[int] = None
    #: The face that was edited, as the service boxed it; null when the node chose (#553).
    face_index: Optional[int] = None
    face_box: Optional[list[float]] = None


class ImageEditResponse(BaseModel):
    uri: str
    source_uri: str
    mode: str
    preset: Optional[str] = None
    prompt: Optional[str] = None
    params: dict[str, float]
    expression: dict[str, float]
    source: Optional[str] = None
    matched_terms: list[str] = []
    dataset_id: Optional[uuid.UUID] = None
    device: Optional[str] = None
    elapsed_ms: Optional[int] = None
    face_index: Optional[int] = None
    face_box: Optional[list[float]] = None


class ImageEditJob(BaseModel):
    """A full-mode edit (#548). `state`: queued -> waiting (the 3090 is rendering, training or
    switching; `message` says which) -> running -> done | failed."""
    id: str
    state: str
    message: str
    source_uri: str
    #: Jobs ahead of this one; null once it has started.
    position: Optional[int] = None
    tag: str
    request: dict
    error: Optional[str] = None
    elapsed_s: Optional[float] = None
    #: When done: the result as a capped JPEG data URI, for the "after" pane. Not stored.
    preview: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    #: {"aura": cosine vs the source | null, "reason": why it is null}. AuraFace, as the
    #: identity harness scores; the console shows it before anything is saved.
    identity: Optional[dict] = None
    prompt: Optional[str] = None
    seed: Optional[int] = None
    saved: list[dict] = []
    #: The box that ran (or is running) it: the standing service or the main 3090 (#570).
    worker: Optional[str] = None
    #: The face it was scoped to, in the source's pixels; null for the whole frame (#569).
    face_box: Optional[list[float]] = None


class ImageEditJobSave(BaseModel):
    dataset_id: Optional[uuid.UUID] = None


class HeadAnglePreset(BaseModel):
    name: str
    label: str
    yaw: float
    pitch: float
    #: Always "full" since #569 (Qwen); was "face" within face_limit_deg (LivePortrait).
    route: str = "full"


class ExpressionPreset(BaseModel):
    """An expression button (#569). The words are the image-edit service's; this is the name
    to send as `preset` with mode "full", and the label to show."""
    name: str
    label: str


class ImageEditFacesRequest(BaseModel):
    source_uri: str = Field(..., min_length=1, max_length=1000)


class DetectedFace(BaseModel):
    index: int
    #: [x1, y1, x2, y2] in the source image's pixels (upright, EXIF orientation applied).
    box: list[float]
    width: float


class ImageEditFaces(BaseModel):
    #: The size the boxes are measured against -- the source's, upright. The console scales
    #: from this to the size it draws the image at, never from the image file's own header.
    width: int
    height: int
    faces: list[DetectedFace]
    #: The face an edit that names none will change; null with no faces.
    default_index: Optional[int] = None


class EditAxis(BaseModel):
    key: str
    label: str
    min: float
    max: float
    step: float
    group: str


class EditPreset(BaseModel):
    name: str
    label: str
    expression: dict[str, float]


class EditPresets(BaseModel):
    mode: str
    presets: list[EditPreset]
    axes: list[EditAxis]
    #: The head-angle section (#548): presets routed by angle, and where the routing splits.
    head_angles: list[HeadAnglePreset] = []
    #: The expression buttons, as Qwen instructions (#569).
    expressions: list[ExpressionPreset] = []
    #: 0 since #569: nothing routes to LivePortrait.
    face_limit_deg: float = 20
    max_yaw: float = 90
    max_pitch: float = 45
