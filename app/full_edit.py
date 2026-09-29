"""Full-mode image edits: Qwen-Image-Edit on the 3090, as asynchronous jobs (wanly-console#548).

WHY A JOB AND NOT A CALL. Face mode (app/face_edit.py) is ~1 s on the 2070 and answers inline.
Full mode needs the 3090's card to itself: ~20 GB of Qwen cannot sit beside a render that holds
~23 of 24 GB. So the box is switched into EDIT MODE first, through the same relay the Workers
page uses for caption mode (POST {worker}:8081/mode) -- and that switch lets the segment in
flight FINISH, which is up to ~27 minutes. No HTTP request can wait that long honestly, so the
console gets a job id back at once and polls it, and can say "3090 is rendering; edit queued"
while it waits instead of a spinner indistinguishable from a hang.

ONE AT A TIME, IN ORDER. One card, one Qwen. A single worker task drains the queue FIFO; the
position in it is part of every status, like the caption queue's (app/caption_queue.py).

HANDING THE CARD BACK. When the queue empties, and after IMAGE_EDIT_RETURN_GRACE_S with nothing
new (so a run of edits pays for one switch, not one each), the box is put back in the mode it
was in before the first edit -- normally render, and the queued renders resume. The box also
hands itself back after ten idle minutes (wanly-gpu-docker control.py), which covers this
process restarting mid-queue.

A TRAINING RUN IS NEVER INTERRUPTED. Edit mode stops the trainer the way caption mode does, and
a stopped trainer is a run destroyed. While the box reports one, jobs wait and say so.

NOTHING IS SAVED UNTIL SAVE. A finished job holds its PNG here, in memory, with the AuraFace
score against the source; the console shows both, and POST .../save writes it as a NEW image --
the same "never overwrite, the user decides" rule as face mode. In memory on purpose: a result
nobody saves within IMAGE_EDIT_JOB_TTL_S is a draft, and drafts do not belong in the bucket. An
API restart loses unsaved results, which costs one re-run.
"""
from __future__ import annotations

import asyncio
import base64
import logging
import time
import uuid
from dataclasses import dataclass, field

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

#: Past this, a head turn is out of LivePortrait's reach and goes to full mode (#547/#548).
#: LivePortrait's own range is ±20 on yaw and pitch; within it, face mode is instant and warps
#: pixels rather than regenerating them, which is the identity-safe choice.
FACE_LIMIT_DEG = 20.0
MAX_YAW = 90.0
MAX_PITCH = 45.0

#: Head-angle presets: name -> (label, yaw, pitch). DEGREES, IN THE IMAGE'S DIRECTIONS: negative
#: yaw turns the face toward the LEFT EDGE OF THE PICTURE (the viewer's left, the subject's
#: right) -- what phase 1's "Turn head left" (rotate_yaw -12) already does, checked on
#: Kelly-2000 sel_008 -- and positive pitch raises the chin. The console routes each by angle:
#: within FACE_LIMIT_DEG to face mode, beyond it to full mode.
HEAD_ANGLES: dict[str, tuple[str, float, float]] = {
    "look_left": ("Look left", -20.0, 0.0),
    "look_right": ("Look right", 20.0, 0.0),
    "three_quarter_left": ("Three-quarter left", -45.0, 0.0),
    "three_quarter_right": ("Three-quarter right", 45.0, 0.0),
    "profile_left": ("Profile left", -90.0, 0.0),
    "profile_right": ("Profile right", 90.0, 0.0),
    "look_up": ("Look up", 0.0, 30.0),
    "look_down": ("Look down", 0.0, -30.0),
}


def route(yaw: float, pitch: float) -> str:
    """"face" when LivePortrait can do it, "full" when it needs Qwen."""
    return "face" if max(abs(yaw), abs(pitch)) <= FACE_LIMIT_DEG else "full"


def face_params(yaw: float, pitch: float) -> dict[str, float]:
    """A face-routed head angle as LivePortrait's numbers. Yaw carries over as is; PITCH IS
    NEGATED, because the node's rotate_pitch > 0 lowers the chin (checked on sel_008: +15 looks
    down) while a head angle's pitch > 0 raises it."""
    out = {}
    if yaw:
        out["rotate_yaw"] = float(yaw)
    if pitch:
        out["rotate_pitch"] = float(-pitch)
    return out


class FullEditError(Exception):
    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


#: Job states, in order. `waiting` is the box's turn not having come yet: rendering (the
#: segment in flight finishes first), training, or mid-switch. `message` says which.
STATES = ("queued", "waiting", "running", "done", "failed")


@dataclass
class Job:
    id: str
    source_uri: str
    request: dict
    #: What the service is asked to do, for the record and the file name: a preset, "angle",
    #: or "instruction".
    tag: str
    state: str = "queued"
    message: str = "queued"
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None
    #: The finished PNG, held until saved or expired. Never sent in a status.
    result: bytes | None = None
    preview: str | None = None
    meta: dict = field(default_factory=dict)
    saved: list[dict] = field(default_factory=list)


class FullEditQueue:
    def __init__(self) -> None:
        self.jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._wake = asyncio.Event()
        self._task: asyncio.Task | None = None
        #: The mode the box was in before this queue switched it, to put back afterwards.
        self._restore_mode: str | None = None

    # ------------------------------------------------------------------ public

    def submit(self, source_uri: str, source: bytes, request: dict, tag: str) -> Job:
        self._expire()
        job = Job(id=uuid.uuid4().hex, source_uri=source_uri, request=request, tag=tag)
        job.meta["_source"] = source
        self.jobs[job.id] = job
        self._order.append(job.id)
        self._wake.set()
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run())
        logger.info("full edit %s queued (%s) for %s, %d ahead", job.id, tag, source_uri,
                    self.position(job) or 0)
        return job

    def get(self, job_id: str) -> Job | None:
        self._expire()
        return self.jobs.get(job_id)

    def position(self, job: Job) -> int | None:
        """How many jobs are ahead of this one; None once it has started."""
        if job.state not in ("queued", "waiting"):
            return None
        return max(0, self._order.index(job.id)) if job.id in self._order else None

    # ---------------------------------------------------------------- the loop

    def _expire(self) -> None:
        cutoff = time.time() - settings.image_edit_job_ttl_s
        for jid in [j.id for j in self.jobs.values()
                    if j.finished_at and j.finished_at < cutoff]:
            self.jobs.pop(jid, None)

    async def _run(self) -> None:
        while True:
            while self._order:
                job = self.jobs.get(self._order[0])
                if job is not None:
                    await self._one(job)
                self._order.pop(0)
            # Queue empty: give the card back after a grace, unless more work arrives first.
            self._wake.clear()
            try:
                await asyncio.wait_for(self._wake.wait(),
                                       timeout=settings.image_edit_return_grace_s)
                continue
            except asyncio.TimeoutError:
                pass
            if self._order:
                continue
            await self._hand_back()
            return

    async def _one(self, job: Job) -> None:
        try:
            await self._box_ready(job)
            job.state, job.message, job.started_at = "running", "editing on the 3090", time.time()
            out = await _edit(job.meta["_source"], job.request)
            png = base64.b64decode(out["image"])
            job.result = png
            # The service sends a capped JPEG beside the PNG (this API has no image library);
            # a service that did not is shown the PNG itself.
            job.preview = ("data:image/jpeg;base64," + out["preview"] if out.get("preview")
                           else "data:image/png;base64," + out["image"])
            job.meta.update({k: out.get(k) for k in (
                "prompt", "seed", "steps", "denoise", "lora", "checkpoint", "identity",
                "width", "height", "vram_peak_mib", "timings_ms")})
            job.state, job.message = "done", "done"
        except FullEditError as e:
            job.state, job.error, job.message = "failed", e.detail, e.detail
        except Exception as e:                      # noqa: BLE001 -- recorded on the job
            logger.exception("full edit %s failed", job.id)
            job.state, job.error, job.message = "failed", f"{type(e).__name__}: {e}", "failed"
        finally:
            job.meta.pop("_source", None)
            job.finished_at = time.time()
            logger.info("full edit %s %s in %.0fs%s", job.id, job.state,
                        job.finished_at - job.created_at,
                        f": {job.error}" if job.error else
                        f", aura {(job.meta.get('identity') or {}).get('aura')}")

    async def _box_ready(self, job: Job) -> None:
        """Wait until the 3090 is in edit mode with image-edit answering, asking for the switch
        if it is not. Every wait says why on the job."""
        deadline = time.time() + settings.image_edit_switch_timeout_s
        asked = False
        while True:
            h = await _health()
            services = {s.get("group") or s.get("name"): s for s in h.get("services", [])}
            mode, pending = h.get("mode"), h.get("pending_mode")
            if "image-edit" not in (h.get("equipped") or []):
                raise FullEditError(
                    503, f"{settings.image_edit_worker} is not equipped for full-mode edits "
                         f"(no image-edit in its SERVICES); redeploy it")
            svc = services.get("image-edit") or {}
            if mode == "edit" and not pending and svc.get("ready"):
                return
            trainer = services.get("lora-trainer") or {}
            if trainer.get("training") and mode != "edit":
                job.state, job.message = "waiting", (
                    f"{settings.image_edit_worker} is training a LoRA; edit queued until it ends")
            elif mode != "edit" and not pending:
                if h.get("mode_error") and asked:
                    raise FullEditError(
                        503, f"{settings.image_edit_worker} could not enter edit mode: "
                             f"{h['mode_error']}")
                if not asked:
                    if self._restore_mode is None:
                        self._restore_mode = mode or "ltx-engine"
                    await _set_mode("edit")
                    asked = True
                job.state, job.message = "waiting", _switch_message(h)
            else:
                job.state, job.message = "waiting", _switch_message(h)
            if time.time() > deadline:
                raise FullEditError(
                    504, f"{settings.image_edit_worker} did not reach edit mode within "
                         f"{settings.image_edit_switch_timeout_s // 60} minutes")
            await asyncio.sleep(settings.image_edit_poll_s)

    async def _hand_back(self) -> None:
        mode, self._restore_mode = self._restore_mode, None
        if not mode or mode == "edit":
            return
        try:
            await _set_mode(mode)
            logger.info("full edit queue empty: %s back to %s", settings.image_edit_worker, mode)
        except Exception as e:                      # noqa: BLE001
            # The box hands itself back after its own idle timeout; this is not the only path.
            logger.warning("could not put %s back in %s mode: %s (it will return by itself "
                           "after its idle timeout)", settings.image_edit_worker, mode, e)


def _switch_message(h: dict) -> str:
    who = settings.image_edit_worker
    render = next((s for s in h.get("services", []) if s.get("name") == "ltx-engine-api"), {})
    if h.get("pending_mode") == "edit" and (render.get("running") or 0):
        return f"{who} is rendering; edit queued (the segment in flight finishes first)"
    if h.get("pending_mode") == "edit":
        return f"{who} is switching to edit mode"
    return f"{who} is getting ready for the edit"


# ----------------------------------------------------------------------- the box


def _control_url() -> str:
    return f"http://{settings.image_edit_worker}:{settings.worker_control_port}"


async def _health() -> dict:
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(f"{_control_url()}/health")
        # A box mid-switch or with a service deliberately off answers 503 with the truth.
        return r.json()
    except Exception as e:
        raise FullEditError(503, f"{settings.image_edit_worker} did not answer on "
                                 f":{settings.worker_control_port} ({e!r})") from e


async def _set_mode(mode: str) -> None:
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(f"{_control_url()}/mode", json={"mode": mode})
    except Exception as e:
        raise FullEditError(503, f"{settings.image_edit_worker} did not accept the mode "
                                 f"switch ({e!r})") from e
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail")
        except Exception:                           # noqa: BLE001
            detail = r.text[:300]
        raise FullEditError(503, f"{settings.image_edit_worker} refused {mode} mode: {detail}")
    logger.info("asked %s for %s mode", settings.image_edit_worker, mode)


async def _edit(source: bytes, request: dict) -> dict:
    url = (settings.image_edit_url or "").strip().rstrip("/")
    if not url:
        raise FullEditError(503, "no image-edit service is configured (image_edit_url is empty)")
    body = {"image": base64.b64encode(source).decode(), **request}
    try:
        async with httpx.AsyncClient(timeout=settings.image_edit_timeout_s) as client:
            r = await client.post(f"{url}/edit", json=body)
    except httpx.TimeoutException as e:
        raise FullEditError(504, f"image-edit did not answer within "
                                 f"{settings.image_edit_timeout_s}s") from e
    except httpx.HTTPError as e:
        raise FullEditError(503, f"image-edit unreachable at {url}: {e!r}") from e
    if r.status_code != 200:
        try:
            detail = r.json().get("detail")
        except Exception:                           # noqa: BLE001
            detail = None
        detail = detail if isinstance(detail, str) else (r.text[:300] or r.reason_phrase)
        code = 422 if r.status_code in (400, 413, 422) else 502
        raise FullEditError(code, f"image-edit: {detail}")
    return r.json()


queue = FullEditQueue()
