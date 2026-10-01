"""Wire shapes for building a character sheet (wanly-console#582, app/sheet_gen.py)."""
import uuid
from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.ltx import LtxCharacterResponse, S3Uri


class SheetGenerateRequest(BaseModel):
    #: ONE photo of her from the Image Repo, full body or most of it, in the outfit
    #: (console#585). It is the turnaround's image 1 -- her build comes from it -- and the
    #: sheet's face panel is auto-cropped from it. There is no body field: the model ignored it.
    photo_uri: S3Uri
    #: What she wears IN THE PHOTO, described: the turnaround keeps it in all three views.
    outfit: str = Field(min_length=1, max_length=600)
    hair: Optional[str] = Field(None, max_length=300)
    #: Pixels around the detected face for the face panel. Default sheet_gen.CROP_PADDING.
    crop_padding: Optional[int] = Field(None, ge=0, le=1024)
    #: The prompt's pronoun. Default: from the character's gender ("man" -> male, else female).
    gender: Optional[Literal["female", "male"]] = None
    #: Who image 1 shows ("young woman"). Default "woman" / "man".
    subject: Optional[str] = Field(None, max_length=60)
    #: How many candidates (one seed each). Ignored when `seeds` is given.
    count: Optional[int] = Field(None, ge=1, le=6)
    seeds: Optional[list[int]] = Field(None, min_length=1, max_length=6)


class SheetCandidate(BaseModel):
    seed: int
    #: The raw 1088x1024 turnaround, and the composed 1536x1024 sheet (jobs bucket, drafts).
    candidate_uri: str
    sheet_uri: str
    #: A capped JPEG of the sheet, for side-by-side comparison.
    preview_uri: Optional[str] = None
    prompt: Optional[str] = None
    model: Optional[str] = None
    files: Optional[dict] = None
    settings: Optional[str] = None
    steps: Optional[int] = None
    cfg: Optional[float] = None
    #: How the face panel was cut: "auto_crop" (from the same photo, console#585) or "centre"
    #: (no detector). Jobs from before #585 say "crop" or "letterbox".
    face_panel: Optional[str] = None
    face_panel_note: Optional[str] = None
    #: The panel's provenance from the service: {source: "same_photo", box, crop, padding,
    #: scale (> 1 = upscaled, soft), detector, det_size, photo_size}.
    face_panel_crop: Optional[dict] = None
    #: A JPEG of the 448x1024 face panel, so the console can show what was cropped.
    face_panel_preview_uri: Optional[str] = None
    #: {"aura": AuraFace cosine of the turnaround vs the photo | null, "reason"}.
    identity: Optional[dict] = None
    width: Optional[int] = None
    height: Optional[int] = None
    timings_ms: Optional[dict] = None
    vram_peak_mib: Optional[int] = None


class SheetJob(BaseModel):
    """queued -> waiting (the box renders, trains or switches; `message` says which) ->
    running ("candidate 2 of 3 on 3090.zero") -> done | failed. Candidates appear as each is
    made, and a failed job keeps the ones it made."""
    id: str
    state: str
    message: str
    character_id: Optional[str] = None
    character_name: Optional[str] = None
    #: The one photo the sheet is built from (a pre-#585 job's face photo).
    photo_uri: Optional[str] = None
    request: dict = {}
    seeds: list[int] = []
    candidates: list[SheetCandidate] = []
    position: Optional[int] = None
    error: Optional[str] = None
    worker: Optional[str] = None
    elapsed_s: Optional[float] = None
    #: Approvals: [{seed, sheet_uri, sheet_id}].
    saved: list[dict] = []


class SheetComposeRequest(BaseModel):
    job_id: str = Field(min_length=1, max_length=64)
    seed: int


class CharacterSheetResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    character_id: Optional[uuid.UUID] = None
    character_name: str
    sheet_uri: str
    candidate_uri: Optional[str] = None
    face_uri: str
    outfit: str
    hair: Optional[str] = None
    body: Optional[str] = None
    gender: Optional[str] = None
    prompt: str
    seed: int
    model: Optional[str] = None
    settings: Optional[str] = None
    files: Optional[dict] = None
    face_panel: Optional[str] = None
    identity: Optional[dict] = None
    job_id: Optional[str] = None
    photo_mode: Optional[str] = None
    face_panel_crop: Optional[dict] = None
    created_at: Optional[datetime] = None


class SheetComposeResponse(BaseModel):
    character: LtxCharacterResponse
    sheet: CharacterSheetResponse


class SheetPresets(BaseModel):
    #: Pre-filled outfit and hair, per pronoun: {"female": {"outfit", "hair"}, "male": ...}.
    defaults: dict[str, dict[str, str]]
    default_count: int
    max_count: int
    #: The face panel's default padding around the detected face, in photo pixels.
    crop_padding: int
