"""What a training request MEANS, and whether it is allowed (wanly-api#352).

The request says WHO: a solo character, or a pair and its two members. Everything else is
derived here, from the registry and the datasets, rather than typed into a dialog:

    triggers + genders      the registry (LtxCharacter), never the request
    each member's images    its kind=character dataset (the request may choose among several)
    the composition set     the pair's kind=composition dataset
    regularization          one kind=regularization pool per gender present
    per-image captions      "<trigger>, <gender>, <body>" from each dataset's stored bodies
    base checkpoint         the render stack's (ltx_stack.py), so a LoRA trains on what it
                            will render on

WHY DERIVED. Every one of these was free text once, and every one went wrong: David's face
got a different trigger in each pair, the "David" LoRA held 14 images of somebody else, no
pair had an image of both people, and a joint run published over its first member's solo
row. A server that derives them cannot be told the wrong thing.

ONE FUNCTION FOR TWO ROUTES. POST /training/preflight returns the plan as a checklist and
POST /training refuses anything with a problem in it. They share `plan_training` so the
Train button and the API cannot disagree about what is allowed -- the console mirrors these
rules for instant feedback, but this is the one that decides.

PROBLEMS BLOCK, WARNINGS DO NOT. Each is {code, message}: the code is for the console to key
a checklist row on, the message is the sentence to show.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app import clips
from app.character_registry import identity_phrase
from app.config import settings
from app.enums import TRAINING_TERMINAL
from app.ltx_stack import LTX_STACK
from app.models import Dataset, LtxCharacter, TrainingJob, training_arch
from app.schemas.training import MAX_DATASET_IMAGES, MIN_DATASET_IMAGES, TrainingCreate

#: Repeats for every identity and composition group: the trainer's long-standing default,
#: which every LoRA so far was trained at. Regularization is sized against it.
CHARACTER_REPEATS = 10
#: Regularization samples per epoch, relative to character samples per epoch. 1.0 is the
#: usual prior-preservation balance: as many generic "woman" steps as "<trigger>, woman"
#: ones, so the class word is pulled back exactly as hard as the trigger pulls it away.
REG_RATIO = 1.0
#: Above this many passes over each character image a run is into memorising the set. A
#: pass is one sight of one image, so it is epochs x repeats: the recipe's 1200 steps over
#: 50 images x 10 repeats is 24 passes, and Kelly-2000 v1 -- the best identity so far -- ran
#: ~30. The first value here (8) counted epochs, not passes, and warned on every good run.
#: A warning, not a refusal -- it is a judgement call.
MAX_PASSES_WARNING = 40
#: How many final captions per group the preview shows.
SAMPLE_CAPTIONS = 5

#: SDXL START-IMAGE LoRAs (#398): the "aio" recipe the trainer reproduces. 8 repeats, and
#: the base the aio LoRAs were trained on (BigaspV2Lustify, on the trainer's models mount
#: under sdxl/). The trainer owns the rest of the recipe.
SDXL_REPEATS = 8

#: VIDEO CLIPS (#411): a member's clips train as their own group beside its stills, through
#: musubi's video dataset (wanly-gpu-docker#189). Each clip yields CLIP_WINDOWS 49-frame
#: windows spread across it, so a clip group's epoch is clips x windows x repeats.
#: PROVISIONAL, like the trainer's 512/49: a 49-frame sample costs far more than a still, and
#: these are set properly by a measured run, not before. 5 x 3 = 15 samples a clip an epoch
#: against a still's 10 -- ten clips beside fifty stills is about a quarter of the epoch.
CLIP_REPEATS = 5
CLIP_WINDOWS = 3
SDXL_BASE_CHECKPOINT = "BigaspV2Lustify"
#: The booru class tag each registry gender trains under in an SDXL PAIR (#407). WD14 strips
#: 1girl/1boy from what it tags, so a solo trigger absorbs "person" generally (the man-bleed);
#: a pair puts the class tag back beside each trigger, the SDXL counterpart of the LTX
#: "<trigger>, <gender>" caption. "person" has no single tag and trains under the trigger alone.
SDXL_CLASS_TAG = {"woman": "1girl", "man": "1boy"}


def sdxl_pair_prefix(triggers: list[str], genders: list[str | None]) -> str:
    """The WD14 prefix an SDXL pair group trains under: its trigger(s), then the class tags of
    the people in frame -- "k3lly, 1girl" for one member, "k3lly, d@vid, 1girl, 1boy" for the
    composition set (two women are "2girls", as booru tags them)."""
    tags = [SDXL_CLASS_TAG[g] for g in genders if g in SDXL_CLASS_TAG]
    if len(tags) == 2 and tags[0] == tags[1]:
        tags = ["2girls" if tags[0] == "1girl" else "2boys"]
    return ", ".join([*triggers, *tags])


@dataclass
class Group:
    """One training group, fully resolved. `captions` pairs 1:1 with `images`."""
    kind: str                      # identity | composition | regularization
    character: str | None
    trigger: str | None
    gender: str | None
    dataset: Dataset | None
    images: list[str] = field(default_factory=list)
    captions: list[str] = field(default_factory=list)
    num_repeats: int = CHARACTER_REPEATS
    #: Training samples each item yields per repeat: 1 for a still, CLIP_WINDOWS for a clip.
    windows: int = 1

    def provenance(self) -> dict:
        ds = self.dataset
        return {"id": str(ds.id) if ds else None, "name": ds.name if ds else None,
                "count": len(self.images)}

    def public(self) -> dict:
        ds = self.dataset
        return {
            "kind": self.kind,
            "character": self.character,
            "trigger": self.trigger,
            "gender": self.gender,
            "dataset_id": str(ds.id) if ds else None,
            "dataset_name": ds.name if ds else None,
            "images": len(self.images),
            "num_repeats": self.num_repeats,
            "windows": self.windows,
            "sample_captions": self.captions[:SAMPLE_CAPTIONS],
        }


@dataclass
class Plan:
    problems: list[dict] = field(default_factory=list)
    warnings: list[dict] = field(default_factory=list)
    groups: list[Group] = field(default_factory=list)
    members: list[LtxCharacter] = field(default_factory=list)
    steps: int = 0
    base_checkpoint: str = LTX_STACK["checkpoint"]
    arch: str = "ltx"
    #: The identity groups' repeats -- what one epoch shows each character image.
    character_repeats: int = CHARACTER_REPEATS

    def problem(self, code: str, message: str) -> None:
        self.problems.append({"code": code, "message": message})

    def warn(self, code: str, message: str) -> None:
        self.warnings.append({"code": code, "message": message})

    @property
    def ok(self) -> bool:
        return not self.problems

    @property
    def samples_per_epoch(self) -> int:
        return sum(len(g.images) * g.num_repeats * g.windows for g in self.groups)

    @property
    def passes_per_image(self) -> float:
        """How many times each CHARACTER image is seen over the run, at batch size 1.

        steps / samples-per-epoch is the epoch count; each epoch shows an identity image
        num_repeats times. Regularization is in the denominator, so adding it halves this
        for the same steps -- which is exactly what the number is for spotting.
        """
        spe = self.samples_per_epoch
        if not spe:
            return 0.0
        return round(self.steps / spe * self.character_repeats, 2)

    def public(self) -> dict:
        return {
            "ok": self.ok,
            "problems": self.problems,
            "warnings": self.warnings,
            "groups": [g.public() for g in self.groups],
            "steps": self.steps,
            "samples_per_epoch": self.samples_per_epoch,
            "passes_per_image": self.passes_per_image,
            "base_checkpoint": self.base_checkpoint,
            "arch": self.arch,
        }


def _same(a: str | None, b: str | None) -> bool:
    """Owner names compare CASE-INSENSITIVELY: "david" typed on a dataset is David."""
    return (a or "").casefold() == (b or "").casefold() and a is not None and b is not None


def _final_caption(prefix: str, body: str | None) -> str:
    body = (body or "").strip()
    return f"{prefix}, {body}" if body else prefix


def _captions_for(body: TrainingCreate, prefix: str, ds: Dataset) -> list[str]:
    """Every image's final caption. trigger_only ignores the stored bodies entirely.

    SDXL (#398): the bare trigger. The trainer re-captions every image with WD14 tags at
    stage time -- the stored bodies are qwen sentences written for LTX, and this base was
    trained and is prompted in booru tags -- so this is a placeholder the preview shows, not
    what trains."""
    if body.arch == "sdxl":
        return [prefix for _ in ds.images]
    if body.caption_mode == "trigger_only":
        return [prefix for _ in ds.images]
    return [_final_caption(prefix, (ds.captions or {}).get(u)) for u in ds.images]


async def plan_training(db: AsyncSession, body: TrainingCreate) -> Plan:
    """Resolve a request into groups and captions, collecting every problem on the way.

    Never raises for a bad configuration: a checklist that stops at the first failure makes
    somebody fix things one refusal at a time. Everything that can be checked is.
    """
    plan = Plan(steps=body.steps, arch=body.arch)
    sdxl = body.arch == "sdxl"
    if sdxl:
        plan.character_repeats = SDXL_REPEATS
        plan.base_checkpoint = SDXL_BASE_CHECKPOINT
        plan.warn("sdxl_wd14",
                  "SDXL: the trainer captions every image with WD14 tags (trigger first"
                  + (", then the class tag — 1girl / 1boy" if body.mode == "pair" else "")
                  + ") and ignores the stored captions. No regularization; not published to "
                  "the character — download it from the run.")
    if body.base_checkpoint:
        name = body.base_checkpoint.removesuffix(".safetensors")
        plan.base_checkpoint = name
        if not sdxl and name != LTX_STACK["checkpoint"]:
            plan.warn("base_differs",
                      f"training against {name!r}; renders use {LTX_STACK['checkpoint']!r}. A "
                      f"LoRA fits the base it was trained on -- compare in the real pipeline.")
    chars = {c.name: c for c in (await db.execute(select(LtxCharacter))).scalars().all()}
    datasets = list((await db.execute(select(Dataset))).scalars().all())
    by_id = {d.id: d for d in datasets}

    # ---- who
    row = chars.get(body.character)
    member_names: list[str] = []
    if body.mode == "solo":
        if body.members:
            plan.problem("members_in_solo", "a solo run has no members — did you mean a pair?")
        if row is None:
            plan.problem("unknown_character",
                         f"{body.character!r} is not a registered character — register it "
                         f"first, with its trigger and gender")
        elif (row.kind or "solo") == "pair":
            plan.problem("solo_on_pair",
                         f"{body.character!r} is a pair; train it as a pair, not solo")
        else:
            member_names = [body.character]
    else:
        if row is not None and (row.kind or "solo") == "solo":
            plan.problem("pair_name_is_solo",
                         f"{body.character!r} is a solo character; a pair needs its own name "
                         f"(e.g. DavidKelly-2026) so it never publishes over a person's row")
        member_names = list(body.members or (row.members if row is not None else None) or [])
        if row is not None and row.members and body.members and \
                list(body.members) != list(row.members):
            plan.problem("members_mismatch",
                         f"{body.character!r} is registered as {' + '.join(row.members)}, "
                         f"not {' + '.join(body.members)}")
        if len(member_names) != 2 or len(set(member_names)) != 2:
            plan.problem("pair_members", "a pair needs exactly two different members")
            member_names = list(dict.fromkeys(member_names))[:2]
        if body.character in member_names:
            plan.problem("pair_is_member", "a pair cannot be one of its own members")
            member_names = [m for m in member_names if m != body.character]

    for name in member_names:
        m = chars.get(name)
        if m is None:
            if body.mode == "pair":  # solo already said this
                plan.problem("member_unknown",
                             f"member {name!r} is not a registered character — register it "
                             f"with its trigger and gender first")
            continue
        if body.mode == "pair" and (m.kind or "solo") != "solo":
            plan.problem("member_not_solo", f"member {name!r} is itself a pair")
            continue
        if not m.trigger or not m.gender:
            plan.problem("trigger_missing",
                         f"{name!r} needs both a trigger and a gender in the registry — the "
                         f"caption is \"<trigger>, <gender>\" and binds to nothing without both")
            continue
        plan.members.append(m)

    triggers = [m.trigger for m in plan.members]
    if len(set(triggers)) != len(triggers):
        plan.problem("duplicate_trigger",
                     "both members have the same trigger — one caption cannot anchor two faces")

    # ---- which datasets
    for key in (body.datasets or {}):
        if key not in member_names:
            plan.problem("dataset_for_non_member",
                         f"a dataset was chosen for {key!r}, who is not in this run")
    used: dict = {}

    def _use(ds: Dataset, label: str) -> bool:
        if ds.id in used:
            plan.problem("dataset_reused",
                         f"{ds.name!r} is used by both {used[ds.id]} and {label} — give each "
                         f"group its own set")
            return False
        used[ds.id] = label
        return True

    clip_groups: list[Group] = []
    for m in plan.members:
        chosen = (body.datasets or {}).get(m.name)
        if chosen is not None:
            ds = by_id.get(chosen)
            if ds is None:
                plan.problem("dataset_not_found", f"the dataset chosen for {m.name!r} does not exist")
                continue
            if ds.kind != "character" or not _same(ds.character, m.name):
                plan.problem("dataset_wrong_owner",
                             f"{ds.name!r} is not a character set owned by {m.name!r} "
                             f"(it is {_describe_owner(ds)})")
                continue
        else:
            own = [d for d in datasets if d.kind == "character" and _same(d.character, m.name)]
            if not own:
                plan.problem("dataset_missing",
                             f"{m.name!r} owns no character dataset — assign one on the "
                             f"Datasets page")
                continue
            if len(own) > 1:
                plan.problem("dataset_ambiguous",
                             f"{m.name!r} owns {len(own)} character datasets "
                             f"({', '.join(sorted(d.name for d in own))}); choose one")
                continue
            ds = own[0]
        if not _use(ds, m.name):
            continue
        # SDXL: the trigger alone leads the WD14 tags, as aio's captions did -- no gender word.
        # An SDXL pair adds the class tag (#407): see SDXL_CLASS_TAG.
        if sdxl:
            prefix = (sdxl_pair_prefix([m.trigger], [m.gender]) if body.mode == "pair"
                      else m.trigger)
        else:
            prefix = identity_phrase(m.trigger, m.gender)
        # STILLS AND CLIPS SPLIT (#411): the stills are the identity group, exactly as before;
        # the clips, if any, are a group of their own -- the trainer reads them as video.
        # _captions_for keeps captions paired with images, so it is filtered after.
        caps = dict(zip(ds.images, _captions_for(body, prefix, ds)))
        stills = [u for u in ds.images if not clips.is_clip(u)]
        moving = [u for u in ds.images if clips.is_clip(u)]
        g = Group(kind="identity", character=m.name, trigger=m.trigger, gender=m.gender,
                  dataset=ds, images=stills, captions=[caps[u] for u in stills],
                  num_repeats=plan.character_repeats)
        _check_character_set(plan, ds, body.allow_low_scores)
        plan.groups.append(g)
        if moving and sdxl:
            plan.warn("sdxl_ignores_clips",
                      f"{ds.name!r}: {len(moving)} clip(s) left out — SDXL trains stills only")
        elif moving:
            clip_groups.append(Group(
                kind="clip", character=m.name, trigger=m.trigger, gender=m.gender, dataset=ds,
                images=moving, captions=[caps[u] for u in moving], num_repeats=CLIP_REPEATS,
                windows=CLIP_WINDOWS))

    # ---- the composition set (pairs only)
    if body.mode == "solo":
        if body.composition_dataset_id:
            plan.problem("composition_in_solo", "a solo run has no composition set")
    else:
        comp = None
        if body.composition_dataset_id:
            comp = by_id.get(body.composition_dataset_id)
            if comp is None:
                plan.problem("dataset_not_found", "the chosen composition dataset does not exist")
            elif comp.kind != "composition" or not _same(comp.character, body.character):
                plan.problem("dataset_wrong_owner",
                             f"{comp.name!r} is not a composition set owned by "
                             f"{body.character!r} (it is {_describe_owner(comp)})")
                comp = None
        elif not body.allow_no_composition:
            own = [d for d in datasets
                   if d.kind == "composition" and _same(d.character, body.character)]
            if len(own) == 1:
                comp = own[0]
            elif len(own) > 1:
                plan.problem("composition_ambiguous",
                             f"{body.character!r} owns {len(own)} composition datasets "
                             f"({', '.join(sorted(d.name for d in own))}); choose one")
            else:
                plan.problem("composition_missing",
                             f"{body.character!r} has no composition dataset — without images "
                             f"of both people together the LoRA holds one face and drops the "
                             f"other. Build one, or acknowledge training without it.")
        if comp is None and body.allow_no_composition:
            plan.warn("no_composition",
                      "training a pair with no composition set: the LoRA has never seen the "
                      "two people in one frame, and tends to hold one face and drop the other")
        if comp is not None and _use(comp, "the composition set"):
            if len(plan.members) == 2 and sdxl:
                prefix = sdxl_pair_prefix([m.trigger for m in plan.members],
                                          [m.gender for m in plan.members])
            elif len(plan.members) == 2:
                prefix = " and ".join(identity_phrase(m.trigger, m.gender)
                                      for m in plan.members)
            else:
                prefix = body.character  # unreachable in a valid plan; keeps previews sane
            comp_caps = dict(zip(comp.images, _captions_for(body, prefix, comp)))
            comp_stills = [u for u in comp.images if not clips.is_clip(u)]
            if len(comp_stills) != len(comp.images):
                plan.warn("composition_clips",
                          f"{comp.name!r}: {len(comp.images) - len(comp_stills)} clip(s) left "
                          f"out — the composition set trains on stills")
            plan.groups.append(Group(
                kind="composition", character=body.character, trigger=None, gender=None,
                dataset=comp, images=comp_stills,
                captions=[comp_caps[u] for u in comp_stills],
                # The identity groups' repeats: CHARACTER_REPEATS for LTX, as always, and
                # aio's 8 for an SDXL pair (#407).
                num_repeats=plan.character_repeats))

    # ---- regularization, one pool per gender present
    genders = list(dict.fromkeys(m.gender for m in plan.members if m.gender))
    reg_groups: list[Group] = []
    if sdxl:
        # aio had none, and the trainer's SDXL path refuses one. Said in sdxl_wd14.
        genders = []
    elif not body.regularization:
        plan.warn("no_regularization",
                  "no regularization pool: identity trains strongest this way, but "
                  + " and ".join(repr(g) for g in genders)
                  + " may drift toward this character (other people in frame take the face)")
        genders = []
    for gender in genders:
        pools = [d for d in datasets if d.kind == "regularization" and d.reg_class == gender]
        if not pools:
            plan.problem("regularization_missing",
                         f"there is no regularization pool for {gender!r} — without one the "
                         f"word {gender!r} drifts toward this character. Create a "
                         f"regularization dataset with class {gender!r} and generate it.")
            continue
        # The largest: a pool is built once per gender and reused by every run, and more
        # distinct people is the only thing a pool is for.
        pool = max(pools, key=lambda d: (len(d.images), d.name))
        if not _use(pool, f"the {gender} regularization pool"):
            continue
        reg_groups.append(Group(
            kind="regularization", character=None, trigger=None, gender=gender, dataset=pool,
            images=list(pool.images),
            captions=[_final_caption(gender, (pool.captions or {}).get(u))
                      for u in pool.images]))

    # ---- per-group checks that apply to every set
    # No caption check (#365): a blank caption IS the standard. _final_caption turns it into
    # the bare phrase -- "<trigger>, <gender>", the pair phrase, or the class word -- which is
    # what identity trains best under (Kelly-2000 v1/v5 bare vs v2's descriptive captions,
    # which lost her). Only props get typed. Requiring a body forced a filler word ("photo")
    # onto every plain image.
    for g in plan.groups + reg_groups:
        _check_common(plan, g)
    # After the checks: a clip group has no 8-item floor (#411) -- three good clips are three
    # good clips -- and its items are the same set's, already checked for duplicates above.
    # Placed after composition, before regularization, which is sized against all of them.
    plan.groups.extend(clip_groups)

    # ---- regularization repeats, sized against what the character groups contribute
    character_samples = sum(len(g.images) * g.num_repeats * g.windows for g in plan.groups)
    for g in reg_groups:
        share = character_samples * REG_RATIO / len(reg_groups)
        g.num_repeats = max(1, round(share / len(g.images))) if g.images else 1
    plan.groups.extend(reg_groups)

    # Not for SDXL: aio's 12 epochs x 8 repeats is 96 passes, and is the recipe.
    if not sdxl and plan.passes_per_image > MAX_PASSES_WARNING:
        plan.warn("passes_high",
                  f"{plan.passes_per_image:g} passes over each character image — above "
                  f"{MAX_PASSES_WARNING} a run is into memorising the set. Fewer steps?")

    # PER ARCH (#402): KimJule's SDXL v1 training does not take LTX v1 -- different models,
    # different run dirs and files. Same expression as the live unique index.
    dupe = (await db.execute(select(TrainingJob).where(
        TrainingJob.character == body.character, TrainingJob.version == body.version,
        training_arch() == body.arch,
        TrainingJob.status.not_in(list(TRAINING_TERMINAL))))).scalars().first()
    if dupe:
        plan.problem("version_taken",
                     f"{body.character} {body.arch.upper()} v{body.version} is already "
                     f"{dupe.status} — cancel it, or pick another version")
    return plan


def _describe_owner(ds: Dataset) -> str:
    if not ds.kind:
        return "unassigned"
    if ds.kind == "regularization":
        return f"a {ds.reg_class or '?'} regularization pool"
    return f"a {ds.kind} set owned by {ds.character!r}"


def _check_character_set(plan: Plan, ds: Dataset, allow_low: bool = False) -> None:
    """A character set must be PROVABLY one person: anchored, scored, every face above the
    floor. The scores are the only evidence the set holds one face -- the 14 stray images
    in the David set would each have scored far below it."""
    if not ds.anchor_uri:
        plan.problem("anchor_missing",
                     f"{ds.name!r} has no anchor face — pick one and score the set")
        return
    scores = ds.scores or {}
    unscored = [u for u in ds.images if u not in scores]
    if unscored:
        plan.problem("scores_missing",
                     f"{ds.name!r}: {len(unscored)} of {len(ds.images)} images have not been "
                     f"scored against the anchor — re-score the set")
    floor = settings.face_cos_floor
    low = [u for u in ds.images if u in scores and (scores[u] is None or scores[u] < floor)]
    if low and allow_low:
        # Acknowledged (console#575): the person vouched these are real photos of this
        # character. Still said out loud, so the run record shows what was waved through.
        plan.warn("score_below_floor_allowed",
                  f"{ds.name!r}: {len(low)} image(s) score below {floor:g} against the anchor "
                  f"(or show no face) — training on them anyway, as verified real photos")
    elif low:
        plan.problem("score_below_floor",
                     f"{ds.name!r}: {len(low)} image(s) score below {floor:g} against the "
                     f"anchor (or show no face) — remove them, or they teach a different face")


def _check_common(plan: Plan, g: Group) -> None:
    ds = g.dataset
    name = ds.name if ds else "?"
    n = len(g.images)
    if n < MIN_DATASET_IMAGES:
        plan.problem("too_few_images",
                     f"{name!r}: {n} images — at least {MIN_DATASET_IMAGES} are needed")
    if n > MAX_DATASET_IMAGES:
        plan.problem("too_many_images",
                     f"{name!r}: {n} images — at most {MAX_DATASET_IMAGES}")
    if len(set(g.images)) != n:
        plan.problem("duplicate_images", f"{name!r} contains duplicates")

