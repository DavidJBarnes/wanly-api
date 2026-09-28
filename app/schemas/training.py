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

from pydantic import BaseModel, Field, field_validator, model_validator

from app.enums import TrainingStatus

#: Below this a run is not worth the GPU hour. p@y worked on 13 and that is the floor anyone has
#: actually proved; 8 is where it stops being arguable.
MIN_DATASET_IMAGES = 8
#: Culling k3llydw from 622 to 50 lifted mean cos 0.558 -> 0.699. More is not better, the extras
#: dilute -- but this is a guard against a mis-click selecting a whole folder, not a quality
#: opinion, so it sits well above the useful range.
MAX_DATASET_IMAGES = 400


#: The fields of the request shape before #352. A body carrying any of them is an old console
#: (or a script) that still thinks it chooses the trigger, gender, images or caption. It is
#: refused with a sentence rather than having those fields silently ignored -- a run that
#: quietly trained under the registry's trigger instead of the one somebody typed would be
#: the worst kind of surprise. Jobs created in the old shape stay claimable and retryable:
#: that is read from the job row, not from a request.
LEGACY_FIELDS = frozenset({
    "trigger", "dataset_id", "dataset_images", "caption", "gender", "identities",
    "second_character", "second_trigger", "second_gender", "second_dataset_images",
    "second_dataset_id", "second_num_repeats", "second_caption",
})


def _path_safe(v: str | None, what: str) -> str | None:
    # It becomes a directory name and an output filename on the trainer.
    if v is not None and ("/" in v or v.startswith(".") or any(c.isspace() for c in v)):
        raise ValueError(f"{what} cannot contain slashes, whitespace, or start with a dot")
    return v


class TrainingCreate(BaseModel):
    """A training request since #352: WHO to train, and the knobs. Nothing else.

    Triggers and genders come from the registry, images from each member's character
    dataset, captions from the datasets' stored bodies, regularization from the pools --
    see app/training_plan.py, which resolves all of it and says what is missing. The same
    body goes to POST /training/preflight and POST /training.

        solo  {mode: "solo", character: "David", ...}
        pair  {mode: "pair", character: "DavidKelly-2026", members: ["David", "Kelly-2026"],
               composition_dataset_id: ..., ...}
    """
    mode: Literal["solo", "pair"]
    #: THE LtxCharacter NAME the LoRA publishes to: the person for solo, the PAIR's own
    #: name for pair (never a member's -- that is the #352 bug where a joint run replaced
    #: what its first member rendered with). `p@y` is correct here; the filename is
    #: sanitised separately (`lora_name`).
    character: str = Field(min_length=1, max_length=64)
    #: Pair only: the two solo characters, group 0 first. Optional when the pair is already
    #: registered with its members.
    members: list[str] | None = Field(default=None, max_length=2)
    #: {member name: dataset id}, only needed when a member owns several character sets.
    datasets: dict[str, uuid.UUID] = Field(default_factory=dict)
    #: Pair only. Omitted: the pair's one composition set, if it has exactly one.
    composition_dataset_id: uuid.UUID | None = None
    #: Pair only: train without a composition set, knowingly. A warning, not a default.
    allow_no_composition: bool = False
    version: int = Field(default=1, ge=1, le=99)
    #: Total training steps across every group, regularization included. The ceiling is a
    #: guard against a typo, not a quality opinion: a pair at the proven passes-per-image
    #: with its regularization legitimately runs well past the old 6000.
    steps: int = Field(default=1200, ge=100, le=30000)
    #: The filename stem the LoRA installs under. ASKED, NOT DERIVED: `p@y` cannot be a
    #: filename, and stripping gives `py` while the file this project renders is `pay_...` --
    #: a human read `@` as `a`, and no rule produces that.
    lora_name: str | None = Field(default=None, max_length=64, pattern=r"^[A-Za-z0-9._-]+$")
    #: Which checkpoints to upload as they are written. "final" is the default: a 650 MB
    #: checkpoint takes ~18 minutes to leave the 3090. Every epoch stays on the trainer and
    #: can be published afterwards.
    publish: Literal["final", "all"] = "final"
    #: HOW IMAGES ARE CAPTIONED. "per_image" trains each image under "<trigger>, <gender>,
    #: <its stored body>". "trigger_only" is the recipe every LoRA before #352 used -- every
    #: image under the bare "<trigger>, <gender>" -- and stored bodies are ignored, so a set
    #: need not be captioned at all.
    #:
    #: Kept because it WON: Kelly-2000 v2 (per-image captions + 1:1 regularization) rendered a
    #: generic woman at every strength where v1, on this recipe, was unmistakably her. The
    #: long captions spread what the trigger had to carry across every other token.
    caption_mode: Literal["per_image", "trigger_only"] = "per_image"
    #: Add a regularization pool per gender. False trains the characters alone, as before
    #: #352: identity stays strongest, and "woman"/"man" may drift toward them (a warning,
    #: not a problem -- it is a trade, not a mistake).
    regularization: bool = True
    #: The base the LoRA trains against, as a bare checkpoint name the trainer resolves under
    #: ltx-2.3/diffusion_models. Default: the render stack's checkpoint. Asked because a
    #: comparison needs it: Kelly-2000 v1 trained on ltx-2.3-22b-dev and v2/v3 on 10Eros, and
    #: whether that swap cost identity cannot be answered without training both. The trainer's
    #: preflight fails the job if the file is not on the box, so a typo costs no GPU time.
    base_checkpoint: str | None = Field(default=None, max_length=128,
                                        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    #: Training seed. Default 42, as every run so far. Varying it with everything else fixed is
    #: how run-to-run noise is measured -- the floor any other comparison must clear.
    seed: int | None = Field(default=None, ge=0, le=2**31 - 1)

    @model_validator(mode="before")
    @classmethod
    def _not_the_legacy_shape(cls, data):
        if isinstance(data, dict):
            legacy = sorted(LEGACY_FIELDS & set(data))
            if legacy:
                raise ValueError(
                    f"the pre-#352 training request is no longer accepted ({', '.join(legacy)}): "
                    f"triggers, genders, images and captions now come from the character "
                    f"registry and the datasets. Send {{mode, character, members?, ...}}.")
        return data

    @field_validator("character")
    @classmethod
    def _no_path_tricks(cls, v: str) -> str:
        return _path_safe(v, "character")


class TrainingProblem(BaseModel):
    code: str
    message: str


class TrainingPlanGroup(BaseModel):
    """One group as the preflight shows it. `sample_captions` are FINAL captions, prefix
    included, exactly as the trainer will write them."""
    kind: Literal["identity", "composition", "regularization"]
    character: str | None = None
    trigger: str | None = None
    gender: str | None = None
    dataset_id: str | None = None
    dataset_name: str | None = None
    images: int
    num_repeats: int
    sample_captions: list[str]


class TrainingPreflight(BaseModel):
    ok: bool
    problems: list[TrainingProblem]
    warnings: list[TrainingProblem]
    groups: list[TrainingPlanGroup]
    steps: int
    samples_per_epoch: int
    passes_per_image: float
    base_checkpoint: str


class TrainingResponse(BaseModel):
    id: uuid.UUID
    character: str
    trigger: str
    version: int
    status: str
    dataset_images: list[str]
    #: The extra groups (#102, #106), each carrying its caption, repeats and provenance
    #: ({character, trigger, gender, caption, images, num_repeats, dataset}). ABSENT for
    #: single-identity runs. The console needs it for the joint image total and the
    #: per-group dataset names.
    identities: list[dict] | None = None
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
    its LoRAs.

    `captions` (#352) pairs 1:1 with them too: group 0's final per-image captions, prefix
    included. None for a job created before #352, which trains under config.caption alone.
    `config.base_checkpoint` names the base model (bare, e.g. 10Eros_v1.5_bf16); absent on
    those older jobs, where the trainer keeps its own default.

    `identities` carries groups 1..N the same way, each
    {character, trigger, gender, caption, captions, kind, num_repeats, dataset_name,
    download_urls}. `kind` is identity | composition | regularization; trigger is None for
    the latter two; `caption` is the legacy single string (the first caption) for a trainer
    that predates `captions`. ABSENT for a run with one group.
    """
    download_urls: list[str]
    captions: list[str] | None = None
    identities: list[dict] | None = None


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
