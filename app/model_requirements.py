"""What a segment needs on disk, and whether a given worker has it.

wanly-console#422. A RunPod pod carries `sulphur_dev_bf16` and nothing else. A pose whose
base model is `10Eros_v1.5_bf16` was handed to it anyway, the start image was uploaded, and
ComfyUI then rejected the graph across three loaders — after the claim, so the segment
failed and the job stalled beside a 3090 that had the file all along.

THE REQUIREMENT IS READ, NEVER INFERRED. Nothing here looks for a substring in a recipe
name, a pose name or a filename. `Segment.ltx_recipe` is a snapshot with explicit keys —
`checkpoint`, `char_lora`, `content_loras[].name` — and those keys are the entire source.
A heuristic that recognises `10Eros_v1.5_bf16` today would fail to recognise `10Eros_v2`
tomorrow, and it would fail as a PASS: the segment would be claimed by a worker that cannot
run it, which is the bug this exists to remove.

THREE CONCEPTS:

    Requirement   Artifact(kind, name), extracted from declared recipe fields.
    Inventory     what a worker reports it can load, per kind.
    Fetchable     kinds a worker can OBTAIN on demand, declared by the worker.

and one rule:

    required - (inventory | artifacts of fetchable kinds) == empty

Fetchability is declared by the worker rather than assumed here. The daemon fetches LoRAs at
claim time already (daemon/lora_sync.py), so LoRAs must not gate; when it learns to fetch
checkpoints (console#423) it will say so in its heartbeat and this gate opens on its own,
with no change here and no coordinated deploy. That seam is the point: the API never holds a
second opinion about what a daemon can do.
"""
from dataclasses import dataclass

from app.ltx_stack import LTX_STACK

CHECKPOINT = "checkpoint"
LORA = "lora"

_EXT = ".safetensors"


@dataclass(frozen=True, order=True)
class Artifact:
    """One file a render needs, and what kind of file it is.

    Frozen and ordered so these can live in sets and be reported in a stable order — a
    blocked segment's message must read the same on every poll.
    """

    kind: str
    name: str

    def __str__(self) -> str:
        return f"{self.kind} '{self.name}'"


def canonical(name: str) -> str:
    """One spelling for a file that is written two ways.

    Recipes and the console store bare stems (`sulphur_dev_bf16`); ComfyUI names files
    (`sulphur_dev_bf16.safetensors`), and the daemon strips the extension before reporting
    so the two agree. Both sides are normalised here anyway, because if that convention ever
    drifts the gate does not fail loudly — it silently matches nothing, and a fleet that
    claims no work looks exactly like an empty queue.
    """
    n = (name or "").strip()
    return n[: -len(_EXT)] if n.endswith(_EXT) else n


def required_artifacts(ltx_recipe: dict | None) -> set[Artifact]:
    """Everything `ltx_recipe` says this segment needs.

    A NULL recipe means "not a recipe render" — a WAN segment, a free-form LTX one, or a CPU
    reprocess carrier. Those declare nothing, so they require nothing and are never gated.
    That is deliberately conservative: this must not withhold work that flows today.

    A recipe with no `checkpoint` key predates per-pose base models and renders on the
    stack's, so the default is resolved HERE rather than at the call sites, where three
    copies of `or LTX_STACK["checkpoint"]` would eventually become two.
    """
    if not ltx_recipe:
        return set()

    out: set[Artifact] = set()

    ckpt = canonical(str(ltx_recipe.get("checkpoint") or ""))
    out.add(Artifact(CHECKPOINT, ckpt or canonical(LTX_STACK["checkpoint"])))

    # "none" is a real, chosen value in this schema — the CHARACTER dropdown offers it
    # (console#413/#416) and the stack's content_lora is the literal string "none". It means
    # no LoRA, so it must never become a requirement for a file called "none.safetensors",
    # which is precisely the bug wanly-gpu-docker#68 fixed downstream.
    for raw in [ltx_recipe.get("char_lora"), *_content_lora_names(ltx_recipe)]:
        name = canonical(str(raw or ""))
        if name and name.lower() != "none":
            out.add(Artifact(LORA, name))

    return out


def _content_lora_names(ltx_recipe: dict) -> list[str]:
    """Content LoRAs are a list of objects, and have been a list of one string before now.

    Tolerant on read because segments rendered under the older shape are still in the
    database and still re-rollable; a KeyError here would take down a job page.
    """
    out: list[str] = []
    for entry in ltx_recipe.get("content_loras") or []:
        if isinstance(entry, dict):
            out.append(str(entry.get("name") or ""))
        elif isinstance(entry, str):
            out.append(entry)
    return out


def unsatisfied(
    required: set[Artifact],
    inventory: dict[str, set[str]],
    fetchable_kinds: list[str] | None,
) -> set[Artifact]:
    """What this worker can neither load nor get. Empty means it may claim the segment.

    A kind ABSENT from `inventory` is unknown, not empty, and does not gate. An older daemon
    reports no checkpoints at all, and starving it of work on an upgrade day would be a
    worse failure than the one being fixed — it would look like a dead queue. `Worker`
    stores NULL for never-reported for the same reason.
    """
    fetchable = {k.strip() for k in (fetchable_kinds or [])}
    return {
        a
        for a in required
        if a.kind not in fetchable
        and a.kind in inventory
        and canonical(a.name) not in {canonical(n) for n in inventory[a.kind]}
    }


def describe(missing: set[Artifact]) -> str:
    """One line naming what is missing, for a person reading a stalled job page."""
    return ", ".join(str(a) for a in sorted(missing))
