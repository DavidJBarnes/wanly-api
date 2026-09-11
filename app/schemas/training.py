"""Wire shapes for character-LoRA training (wanly-api#274).

Three audiences, and they want different things, which is why this is not one model:

  the console   creates a job and reads its progress
  the trainer   claims a job and needs EVERYTHING to run it in one response
  both          list and inspect

The claim response is deliberately self-contained. Same rule as SegmentClaimResponse: a worker
must never look its configuration up for itself, because a worker that cannot look one up cannot
look up a STALE one.
"""
import uuid
from datetime import datetime

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from app.enums import TrainingStatus

#: Below this a run is not worth the GPU hour. p@y worked on 13 and that is the floor anyone has
#: actually proved; 8 is where it stops being arguable.
MIN_DATASET_IMAGES = 8
#: Culling k3llydw from 622 to 50 lifted mean cos 0.558 -> 0.699. More is not better, the extras
#: dilute -- but this is a guard against a mis-click selecting a whole folder, not a quality
#: opinion, so it sits well above the useful range.
MAX_DATASET_IMAGES = 400


class TrainingCreate(BaseModel):
    #: THE LtxCharacter NAME, not a filesystem-safe version of it. `p@y` is correct here.
    #:
    #: This is what the finished LoRA is published against, and it upserts on the name -- so
    #: passing `pay` for a character the system already knows as `p@y` creates a SECOND
    #: character row pointing at the same LoRA, and recipes keep using the old one. The
    #: filename is sanitised separately at upload (`pay_v3_e04.safetensors`), because a LoRA
    #: is served over HTTP and lands in JSON and URLs; the name and the trigger are not.
    character: str = Field(min_length=1, max_length=64)
    trigger: str = Field(min_length=1, max_length=64)
    version: int = Field(default=1, ge=1, le=99)
    #: Either give the images, or name a dataset and let the API resolve them. The dataset is
    #: the normal path now -- a set worth training is a set worth being able to re-open. The
    #: minimum is enforced in the route rather than here, because it applies to whichever of
    #: the two was used.
    dataset_images: list[str] = Field(default_factory=list, max_length=MAX_DATASET_IMAGES)
    dataset_id: uuid.UUID | None = None
    #: One caption for every image. Per-image captions come from the dataset instead, and are
    #: the better answer -- all 13 of p@y's read "p@y, woman" over close-ups, so the trigger
    #: carries close-up framing as part of its identity.
    caption: str | None = Field(default=None, max_length=500)
    #: The console's way of saying the caption: every image is captioned "<trigger>, <gender>".
    #: Explicit because a free caption field was filled in with "man" alone and the trigger
    #: was never learned. When given, it decides the caption.
    gender: Literal["woman", "man", "person"] | None = None
    steps: int = Field(default=1200, ge=100, le=6000)
    #: The filename stem the LoRA installs under. ASKED, NOT DERIVED.
    #:
    #: A LoRA is served over HTTP and lands in JSON and URLs, so `p@y` cannot be a filename --
    #: but stripping the character it cannot carry gives `py`, and the file this project has
    #: actually been rendering with is `pay_v2_e05.safetensors`. A human read `@` as `a`. There
    #: is no rule that produces that: `@`->`a` is a transliteration, and inventing a table for
    #: it generalises badly (k3lly2026 keeps its digits, so `3`->`e` is wrong).
    #:
    #: So it is a field, defaulted to the stripped form and visible in the dialog, rather than
    #: a guess made silently at upload time.
    lora_name: str | None = Field(default=None, max_length=64,
                                  pattern=r"^[A-Za-z0-9._-]+$")
    #: Which checkpoints to upload as they are written. "final" is the default: a 650 MB
    #: checkpoint takes ~18 minutes to leave the 3090 and a five-epoch run's uploads
    #: outlast the training, for epochs that mostly go unused. "all" uploads every one.
    #: Either way every epoch stays on the trainer and can be published afterwards.
    publish: Literal["final", "all"] = "final"
    #: A SECOND identity, making this a JOINT run (#102): one LoRA trained on both
    #: characters' datasets simultaneously, whose group-0 delta is learned in the presence
    #: of group-1's data. This is the structural fix for two-identity interference (#100,
    #: R2: no strength setting recovers two-char identity; two independently-trained deltas
    #: fight in the shared modules). ABSENT means single-identity, which every run before
    #: this is.
    #:
    #: The fields mirror group 0's, with its own images and num_repeats: the two datasets
    #: balance through repeats, not by truncating the smaller set.
    second_character: str | None = Field(default=None, min_length=1, max_length=64)
    second_trigger: str | None = Field(default=None, min_length=1, max_length=64)
    second_gender: Literal["woman", "man", "person"] | None = None
    second_dataset_images: list[str] = Field(default_factory=list, max_length=MAX_DATASET_IMAGES)
    second_dataset_id: uuid.UUID | None = None
    second_num_repeats: int | None = Field(default=None, ge=1, le=100)
    #: A free caption for the second group, used when second_gender is absent. Must name
    #: the second trigger, or the face binds to nothing — the route enforces that.
    second_caption: str | None = Field(default=None, max_length=500)

    @field_validator("character")
    @classmethod
    def _no_path_tricks(cls, v: str) -> str:
        # It becomes a directory name and an output filename on the trainer.
        if "/" in v or v.startswith(".") or any(c.isspace() for c in v):
            raise ValueError("character cannot contain slashes, whitespace, or start with a dot")
        return v

    @field_validator("second_character")
    @classmethod
    def _second_no_path_tricks(cls, v: str | None) -> str | None:
        if v is None:
            return v
        if "/" in v or v.startswith(".") or any(c.isspace() for c in v):
            raise ValueError("second_character cannot contain slashes, whitespace, or start with a dot")
        return v

    @field_validator("dataset_images")
    @classmethod
    def _s3_uris_only(cls, v: list[str]) -> list[str]:
        bad = [x for x in v if not x.startswith("s3://")]
        if bad:
            raise ValueError(f"dataset_images must be s3:// URIs, got {bad[0]!r}")
        if len(set(v)) != len(v):
            # Duplicates train the same image twice under two sel_NNN names, silently
            # reweighting the set.
            raise ValueError("dataset_images contains duplicates")
        return v

    @field_validator("second_dataset_images")
    @classmethod
    def _second_s3_uris_only(cls, v: list[str]) -> list[str]:
        bad = [x for x in v if not x.startswith("s3://")]
        if bad:
            raise ValueError(f"second_dataset_images must be s3:// URIs, got {bad[0]!r}")
        if len(set(v)) != len(v):
            raise ValueError("second_dataset_images contains duplicates")
        return v


class TrainingResponse(BaseModel):
    id: uuid.UUID
    character: str
    trigger: str
    version: int
    status: str
    dataset_images: list[str]
    config: dict
    worker_name: str | None = None
    gpu_name: str | None = None
    progress_log: str | None = None
    step: int | None = None
    total_steps: int | None = None
    error_message: str | None = None
    checkpoints: list | None = None
    output_lora_path: str | None = None
    # [[step, avr_loss], ...] sampled from the trainer's progress bar, for the curve.
    loss_log: list | None = None
    # Every checkpoint the run WROTE, as [{label, step, loss}], published or not. Only the
    # final one is uploaded by default: a 650 MB checkpoint takes ~18 minutes to leave the
    # 3090, and "I often only want 1 or 2 epochs". The rest sit on the trainer until asked
    # for through publish_requests, and `checkpoints` records the ones that arrived.
    epochs: list | None = None
    publish_requests: list | None = None
    # Operator notes, written through the console's own route (wanly-console#484). The
    # trainer's PATCH deliberately cannot reach this: its conditional writes exist so a
    # report that omits a field never blanks a previous one, and a human note deserves the
    # stronger guarantee that no report can blank it at all.
    notes: str | None = None
    thumbnail_uri: str | None = None
    created_at: datetime | None = None
    claimed_at: datetime | None = None
    completed_at: datetime | None = None

    model_config = {"from_attributes": True}


class TrainingClaimResponse(TrainingResponse):
    """What a trainer gets. Everything it needs, resolved.

    `download_urls` pairs 1:1 with dataset_images, in order, so the trainer never needs S3
    credentials -- it fetches through the API's own proxy, the same way the render daemon gets
    its LoRAs. second_* carry the joint group the same way: ABSENT for every single-identity
    run, so a trainer that predates #102 sees nothing.
    """
    download_urls: list[str]
    second_download_urls: list[str] | None = None
    second_caption: str | None = None
    second_num_repeats: int | None = None


class TrainingProgress(BaseModel):
    """A trainer reporting in. Every field optional and written only when present.

    Same conditional-write rule as the worker heartbeat: a report that omits a field must not
    blank what a previous one set, or a trainer that only sends `step` erases the log.
    """
    status: TrainingStatus | None = None
    progress_log: str | None = None
    step: int | None = Field(default=None, ge=0)
    total_steps: int | None = Field(default=None, ge=1)
    error_message: str | None = None
    checkpoints: list | None = None
    output_lora_path: str | None = None
    loss_log: list | None = None
    epochs: list | None = None


class TrainingNotes(BaseModel):
    """Operator notes, whole-field replace.

    NOT part of TrainingProgress deliberately: that model is the trainer's report channel,
    whose conditional writes exist so an omitted field never blanks what came before. A note
    wants the opposite contract -- an explicit write that the reports can never touch -- and
    so it has its own tiny shape and its own route.
    """
    notes: str | None = Field(default=None, max_length=20000)
