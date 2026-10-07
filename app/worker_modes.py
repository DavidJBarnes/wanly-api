"""Which box is in which mode, and where pushed work goes because of it (wanly-api#392).

THE MODEL (wanly-console#572, wanly-gpu-docker#164). Each 3090 is in ONE mode at a time --
`render`, `train`, `motion` or `edit` -- and either box can be in any of them. Renders and
training are PULLED (a box claims what its mode lets it run), so they need nothing from here.
Motion captions and image edits are PUSHED to a URL, and that URL used to be one configured
box. With two symmetric boxes the right box is whichever one is in the mode right now, so it
is found at request time, from the boxes themselves.

ASKED OF THE BOX, NEVER STORED. Same rule as GET /workers/{id}/mode: the container is the only
thing that knows its mode (a curl can flip it without this API hearing), so a column would be a
second copy free to disagree. Each box's /health is read and cached for a few seconds -- a mode
changes on a human action, and a batch of captions must not put the control API in its hot
path.

NOTHING REPORTS -> THE OLD WAY. A deployment whose boxes cannot say their mode (an older
container, or none reachable) keeps the configured URL and its old refusal rules. Only when at
least one box answers with a mode does routing by mode take over.

SPELLINGS. Boxes report `mode_name` in the four-mode spelling and `mode` in the old one
(ltx-engine / caption). A box from before #164 has only `mode`. Both are read, through
canonical(), so every comparison here is between modes, not spellings.

PORTS. A box's /health knows its CONTAINER ports, not the host ports they are published on,
so the URL of a box's captioner or image-edit takes its port from the configured URL
(motion_caption_url / image_description_url, image_edit_url). Every box publishes a service on
the same host port, like worker_control_port.
"""
from __future__ import annotations

import asyncio
import itertools
import logging
import time
from dataclasses import dataclass, field
from urllib.parse import urlsplit

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

MODES = ("render", "train", "motion", "edit")

#: The names modes had before wanly-gpu-docker#164, and other spellings. Mirrors the box's own
#: registry._MODE_ALIASES so the two agree on what a word means.
_ALIASES = {"ltx-engine": "render", "engine": "render",
            "caption": "motion", "image-caption": "motion", "image-description": "motion",
            "motion-caption": "motion",
            "training": "train", "trainer": "train", "lora-trainer": "train",
            "image-edit": "edit", "full-edit": "edit"}

#: The SERVICES group whose readiness a mode's pushed work needs.
SERVICE_FOR = {"motion": "image-description", "edit": "image-edit"}

_TTL_S = 5.0


def canonical(raw: str | None) -> str | None:
    """The mode's one spelling, or None for nothing / not a mode."""
    if not raw:
        return None
    m = str(raw).strip().lower()
    m = _ALIASES.get(m, m)
    return m if m in MODES else None


@dataclass
class BoxState:
    """One box, as its /health described it."""
    name: str
    #: False when the box did not answer at all.
    reachable: bool = False
    mode: str | None = None
    pending: str | None = None
    modes: list[str] = field(default_factory=list)
    equipped: list[str] = field(default_factory=list)
    services: list[dict] = field(default_factory=list)
    gpu: dict | None = None
    last_unload: dict | None = None
    mode_error: str | None = None
    #: The spellings as the box sent them, for the API's own responses.
    raw_mode: str | None = None
    raw_pending: str | None = None

    def service_ready(self, group: str) -> bool | None:
        """Is that SERVICES group running and ready? None when the box lists no such service."""
        found = [s for s in self.services if (s.get("group") or s.get("name")) == group]
        if not found:
            return None
        return all(s.get("ready") for s in found if not s.get("stopped")) and any(
            not s.get("stopped") for s in found)


def parse_health(name: str, body: dict | None) -> BoxState:
    """A /health body as a BoxState. Tolerates every older shape: missing keys are absent."""
    if not isinstance(body, dict):
        return BoxState(name=name)
    mode = canonical(body.get("mode_name")) or canonical(body.get("mode"))
    pending = canonical(body.get("pending_mode_name")) or canonical(body.get("pending_mode"))
    modes = [m for m in (canonical(x) for x in (body.get("modes") or [])) if m]
    return BoxState(
        name=name, reachable=True, mode=mode, pending=pending, modes=modes,
        equipped=list(body.get("equipped") or []),
        services=list(body.get("services") or []),
        gpu=body.get("gpu") if isinstance(body.get("gpu"), dict) else None,
        last_unload=body.get("last_unload") if isinstance(body.get("last_unload"), dict) else None,
        mode_error=body.get("mode_error"),
        raw_mode=body.get("mode"), raw_pending=body.get("pending_mode"))


_CACHE: dict[str, tuple[float, BoxState]] = {}


def forget(name: str | None = None) -> None:
    """Drop the cached state of one box (after asking it to switch), or of all of them."""
    if name is None:
        _CACHE.clear()
    else:
        _CACHE.pop(name.lower(), None)


async def _fetch(name: str) -> dict | None:
    """GET a box's /health body, or None. A degraded box answers 503 with the truth in it."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"http://{name}:{settings.worker_control_port}/health")
        body = r.json()
        return body if isinstance(body, dict) else None
    except Exception:  # noqa: BLE001 - an unreachable box is an answer: it reports nothing
        return None


async def box_state(name: str) -> BoxState:
    key = name.lower()
    hit = _CACHE.get(key)
    now = time.monotonic()
    if hit and now - hit[0] < _TTL_S:
        return hit[1]
    st = parse_health(name, await _fetch(name))
    _CACHE[key] = (now, st)
    return st


async def live_boxes(db) -> list[BoxState]:
    """Every box with a non-offline worker row, read concurrently, in name order."""
    from sqlalchemy import select

    from app.models import Worker
    rows = (await db.execute(select(Worker).where(Worker.status != "offline"))).scalars().all()
    names = sorted({(w.friendly_name or "").strip() for w in rows} - {""}, key=str.lower)
    return list(await asyncio.gather(*(box_state(n) for n in names)))


async def live_boxes_own_session() -> list[BoxState]:
    from app.database import async_session
    async with async_session() as db:
        return await live_boxes(db)


@dataclass
class Pick:
    """Where pushed work for a mode goes right now.

    Exactly one of three: `box` (send it there), `wait` (a reason, shown to the person; nothing
    is sent), or neither with `reporting` False (no box says its mode: the caller's old way).
    """
    mode: str
    box: str | None = None
    wait: str | None = None
    reporting: bool = True


_turns = itertools.count()


def choose(mode: str, boxes: list[BoxState]) -> Pick:
    """Pick a box in `mode` from what the boxes reported. Pure: no I/O.

    A box qualifies when it is in the mode, not switching away from it, and the service the
    mode's work needs is not reported down. Two qualify: take turns. None qualifies: say why,
    in the words that tell the person what to do.
    """
    reporting = [b for b in boxes if b.reachable and b.mode]
    if not reporting:
        return Pick(mode=mode, reporting=False)
    group = SERVICE_FOR.get(mode)
    ready = [b for b in reporting if b.mode == mode and not b.pending
             and (group is None or b.service_ready(group) is not False)]
    if ready:
        return Pick(mode=mode, box=ready[next(_turns) % len(ready)].name)
    switching = [b for b in reporting if b.pending == mode]
    if switching:
        b = switching[0]
        return Pick(mode=mode, wait=f"{b.name} is switching to {mode} mode")
    starting = [b for b in reporting if b.mode == mode and not b.pending]
    if starting:
        b = starting[0]
        return Pick(mode=mode, wait=f"{b.name} is in {mode} mode; its {group} is starting")
    return Pick(mode=mode, wait=no_gpu_in(mode, reporting))


def no_gpu_in(mode: str, boxes: list[BoxState]) -> str:
    """'no GPU in motion mode (3090a: render, 3090b: edit); switch one on the Workers page'."""
    where = ", ".join(f"{b.name}: {b.mode}{f' -> {b.pending}' if b.pending else ''}"
                      for b in boxes if b.mode)
    return (f"no GPU in {mode} mode" + (f" ({where})" if where else "")
            + f"; switch one to {mode} on the Workers page")


async def pick(db, mode: str) -> Pick:
    return choose(mode, await live_boxes(db))


def url_on(box: str, configured_url: str | None, default_port: int) -> str:
    """The URL of a service on `box`, on the port the configured URL uses."""
    port = None
    if configured_url:
        try:
            port = urlsplit(configured_url.strip()).port
        except ValueError:
            port = None
    return f"http://{box}:{port or default_port}"


def motion_url(box: str) -> str:
    return url_on(box, settings.motion_caption_url or settings.image_description_url, 11434)


def edit_url(box: str) -> str:
    return url_on(box, settings.image_edit_url, 8086)
