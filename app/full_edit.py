"""Full-mode image edits: Qwen-Image-Edit, as asynchronous jobs (wanly-console#548, #569, #570).

EVERY EDIT IN THE EDIT DIALOG IS ONE OF THESE NOW (#569). LivePortrait (app/face_edit.py) "drops
every detail"; head angles at any size, the expression presets and the free-text box all come
here. The face-mode endpoints stay for now, but nothing in the console calls them.

WHERE AN EDIT RUNS is "a worker equipped with image-edit", in two shapes. An ALWAYS-ON one
(`image_edit_standing_url`: image-edit running without a mode switch) is preferred whenever its
/health is ok; otherwise the job goes to `image_edit_worker`'s EDIT MODE, below, which pauses its
renders. No box is named in code: which boxes carry image-edit is being re-planned as symmetric
3090 workers, and choosing among them extends `_on_standing` / `_box_ready`. An always-on worker
that is merely busy (its service says it is waiting, e.g. for an A1111 on its card) is waited
for, not fallen back from: the fallback is the one path that pauses renders.

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

CHARACTER-SHEET JOBS SHARE THIS QUEUE (wanly-console#582, app/sheet_gen.py). A sheet job is
N turnaround candidates on the same service, so it needs the same card, the same edit mode and
the same "never interrupt training" rule -- one queue, FIFO with the edits, one hand-back. A
job's `kind` says which; `work` is the sheet's own runner, called with the URL of whichever
worker the job landed on.

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

MAX_YAW = 90.0
MAX_PITCH = 45.0

#: Head-angle presets: name -> (label, yaw, pitch). DEGREES, IN THE IMAGE'S DIRECTIONS: negative
#: yaw turns the face toward the LEFT EDGE OF THE PICTURE (the viewer's left, the subject's
#: right) -- what phase 1's "Turn head left" (rotate_yaw -12) already did, checked on
#: Kelly-2000 sel_008 -- and positive pitch raises the chin. All of them run on Qwen (#569).
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


#: The expression presets (#569), name -> label. THE WORDS LIVE IN THE SERVICE
#: (wanly-gpu-docker image_edit/graph.py EXPRESSIONS), beside the head-angle words and the model
#: they are written for, so retuning one never needs an API deploy; this only lists what the
#: console offers and what a request may name. The names are the old LivePortrait preset names
#: where one existed, so a saved file's `_edit-smile_` tag reads the same across the switch.
EXPRESSIONS: dict[str, str] = {
    "smile": "Smile",
    "big_laugh": "Big laugh",
    "surprised": "Surprised",
    "eyes_closed": "Eyes closed",
    "sad": "Sad",
    "angry": "Angry",
    "serious": "Serious",
    "speaking": "Speaking",
    "look_left": "Eyes left",
    "look_right": "Eyes right",
    "look_up": "Eyes up",
    "look_down": "Eyes down",
}

#: What a request needs from the service beyond #548's angle/instruction. An image-edit image
#: from before #569 IGNORES these fields (pydantic drops unknown keys): an expression beside an
#: angle would come back as the angle alone, a face_box as the whole frame regenerated -- a
#: plausible, wrong result. So the service's /health `features` is checked first, and a box
#: that lacks one fails the job saying "re-pin it" instead.
_FEATURE_FIELDS = ("expression", "face_box", "turnaround")


class FullEditError(Exception):
    def __init__(self, status_code: int, detail: str, unreachable: bool = False):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail
        #: The request never reached the service (connection refused, DNS): safe to send the
        #: edit elsewhere, because it cannot be running.
        self.unreachable = unreachable


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
    #: Which worker ran it: the always-on one's name, or image_edit_worker.
    worker: str | None = None
    #: "edit" (the Edit dialog) or "sheet" (a character sheet's candidates, app/sheet_gen.py).
    kind: str = "edit"
    #: A sheet job's runner: work(job, url) -> dict. None for an edit.
    work: object | None = None
    #: What the service must advertise on /health for this job (see _FEATURE_FIELDS).
    needs: dict = field(default_factory=dict)
    #: A sheet job's hook once it is over, done or failed: on_finish(job) (persists its state).
    on_finish: object | None = None


class FullEditQueue:
    def __init__(self) -> None:
        self.jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._wake = asyncio.Event()
        self._task: asyncio.Task | None = None
        #: The mode the box was in before this queue switched it, to put back afterwards.
        self._restore_mode: str | None = None

    # ------------------------------------------------------------------ public

    def submit(self, source_uri: str, source: bytes, request: dict, tag: str,
               job: Job | None = None) -> Job:
        """Queue an edit -- or `job`, already built (a sheet job, app/sheet_gen.py)."""
        self._expire()
        if job is None:
            job = Job(id=uuid.uuid4().hex, source_uri=source_uri, request=request, tag=tag)
        job.meta["_source"] = source
        self.jobs[job.id] = job
        self._order.append(job.id)
        self._wake.set()
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run())
        logger.info("full edit %s queued (%s %s) for %s, %d ahead", job.id, job.kind, job.tag,
                    job.source_uri, self.position(job) or 0)
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
            if job.kind != "edit":
                await self._place(job)
                job.state, job.message = "done", "done"
                return
            out = await self._place(job)
            _check_echo(out, job.request, job.worker or "image-edit")
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
            logger.info("full edit %s (%s) %s in %.0fs%s", job.id, job.kind, job.state,
                        job.finished_at - job.created_at,
                        f": {job.error}" if job.error else
                        f", aura {(job.meta.get('identity') or {}).get('aura')}")
            if callable(job.on_finish):
                try:
                    await job.on_finish(job)
                except Exception:                   # noqa: BLE001 -- the job is already over
                    logger.exception("full edit %s: on_finish failed", job.id)

    async def _call(self, job: Job, url: str | None) -> dict:
        """Run the job's work against one worker's image-edit service."""
        if job.kind == "edit":
            if url is None:
                return await _edit(job.meta["_source"], job.request)
            return await _edit(job.meta["_source"], job.request, url=url)
        return await job.work(job, (url or settings.image_edit_url or "").strip().rstrip("/"))

    def _running(self, job: Job) -> str:
        return (f"editing on {job.worker}" if job.kind == "edit"
                else f"generating on {job.worker}")

    async def _place(self, job: Job) -> dict:
        """The always-on worker when it is up, else the main 3090's edit mode; then the work."""
        out = await self._on_standing(job)
        if out is None:
            await self._box_ready(job)
            await _require_features(settings.image_edit_url, _needs(job),
                                    settings.image_edit_worker)
            job.worker = settings.image_edit_worker
            job.state, job.message, job.started_at = (
                "running", self._running(job), time.time())
            out = await self._call(job, None)
        return out

    async def _on_standing(self, job: Job) -> dict | None:
        """Run the job on the always-on image-edit worker, or None to fall back to edit mode.

        None when no standing service is configured, when its /health is not ok, or when it
        stops answering before the edit is sent. A standing box that is only BUSY -- A1111
        generating, or its own edit in flight -- is waited for with the reason on the job,
        within the same budget as a mode switch. Once the edit has been SENT there is no
        fallback: a timeout then may be an edit still running, and running it twice is worse
        than failing it.
        """
        url = _standing_url()
        if not url:
            return None
        name = _standing_name(url)
        deadline = time.time() + settings.image_edit_switch_timeout_s
        while True:
            h = await _standing_health(url)
            if h is None:
                logger.info("full edit %s: %s is not healthy; falling back to %s edit mode",
                            job.id, name, settings.image_edit_worker)
                return None
            if h.get("a1111_generating"):
                job.state, job.message = "waiting", (
                    f"{name} busy (A1111 generating); edit queued")
            elif h.get("waiting"):
                job.state, job.message = "waiting", f"{name} busy ({h['waiting']}); edit queued"
            else:
                await _require_features(url, _needs(job), name, health=h)
                job.worker = name
                job.state, job.message, job.started_at = (
                    "running", self._running(job), time.time())
                try:
                    return await self._call(job, url)
                except FullEditError as e:
                    if e.unreachable:
                        logger.warning("full edit %s: %s unreachable (%s); falling back",
                                       job.id, name, e.detail)
                        job.worker, job.started_at = None, None
                        return None
                    if e.status_code == 503 and "generating" in e.detail:
                        # A1111 started between our look and the edit, and outlasted the
                        # service's own wait. Back to waiting, not failed and not fallen back.
                        job.state, job.message = "waiting", (
                            f"{name} busy (A1111 generating); edit queued")
                    else:
                        raise
            if time.time() > deadline:
                raise FullEditError(
                    504, f"{name} stayed busy for {settings.image_edit_switch_timeout_s // 60} "
                         f"minutes (A1111 generating); pause generate-forever to let edits in")
            await asyncio.sleep(settings.image_edit_poll_s)

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


def _needs(job: Job) -> dict:
    """What the feature check reads: an edit's own request, or a sheet job's `needs`."""
    return job.request if job.kind == "edit" else job.needs


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


def _standing_url() -> str:
    return (settings.image_edit_standing_url or "").strip().rstrip("/")


def _standing_name(url: str) -> str:
    """What a job's status calls the always-on worker: its configured name, else its host."""
    from urllib.parse import urlparse
    return (settings.image_edit_standing_name or "").strip() or urlparse(url).hostname or url


async def _standing_health(url: str) -> dict | None:
    """The standing service's /health when it is up and ready for an edit, else None.

    "Ready" is its own `status: ok` (ComfyUI answering). Quick timeout: this is asked before
    every job, and a box that is down must cost a moment, not the job."""
    try:
        async with httpx.AsyncClient(timeout=settings.image_edit_standing_timeout_s) as client:
            r = await client.get(f"{url}/health")
        h = r.json() if r.status_code == 200 else None
    except Exception:                               # noqa: BLE001 -- down is an answer
        return None
    return h if isinstance(h, dict) and h.get("status") == "ok" else None


async def _require_features(url: str, request: dict, who: str,
                            health: dict | None = None) -> None:
    """Refuse a request the service would silently mis-apply (see _FEATURE_FIELDS)."""
    wanted = [f for f in _FEATURE_FIELDS if request.get(f) is not None]
    if not wanted:
        return
    if health is None:
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                health = (await client.get(f"{url.rstrip('/')}/health")).json()
        except Exception as e:                      # noqa: BLE001
            raise FullEditError(503, f"image-edit on {who} did not answer /health ({e!r})") \
                from e
    have = set((health or {}).get("features") or ("angle", "instruction"))
    missing = [f for f in wanted if f not in have]
    if missing:
        raise FullEditError(
            503, f"image-edit on {who} is too old for {' and '.join(missing)} "
                 f"(console#569); re-pin it to a current wanly-gpu-docker image")


def _check_echo(out: dict, request: dict, who: str) -> None:
    """Belt and braces for _require_features: a face-scoped edit must come back saying so."""
    if request.get("face_box") is not None and out.get("face_box") is None:
        raise FullEditError(
            502, f"image-edit on {who} ignored the chosen face (an image from before "
                 f"console#569); nothing was kept -- re-pin it")


async def _edit(source: bytes, request: dict, url: str | None = None) -> dict:
    body = {"image": base64.b64encode(source).decode(), **request}
    return await post_service(url, "/edit", body)


async def post_service(url: str | None, path: str, body: dict) -> dict:
    """POST to an image-edit service, every failure a FullEditError that says which kind."""
    url = (url or settings.image_edit_url or "").strip().rstrip("/")
    if not url:
        raise FullEditError(503, "no image-edit service is configured (image_edit_url is empty)")
    try:
        async with httpx.AsyncClient(timeout=settings.image_edit_timeout_s) as client:
            r = await client.post(f"{url}{path}", json=body)
    except (httpx.ConnectError, httpx.ConnectTimeout) as e:
        # Before TimeoutException, which ConnectTimeout subclasses: never connected means the
        # edit cannot be running, so the caller may send it elsewhere.
        raise FullEditError(503, f"image-edit unreachable at {url}: {e!r}",
                            unreachable=True) from e
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


async def faces(source: bytes) -> dict | None:
    """The face list from the always-on image-edit worker, or None when there is none to ask.

    Only an always-on one: image_edit_worker runs image-edit in edit mode only, and a face list
    is not worth stopping a render for. The route falls back to face-edit's list, whose boxes
    are as good for a crop -- a box is a box, whichever detector drew it."""
    url = _standing_url()
    if not url or await _standing_health(url) is None:
        return None
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(f"{url}/faces",
                                  json={"image": base64.b64encode(source).decode()})
        out = r.json() if r.status_code == 200 else None
    except Exception as e:                          # noqa: BLE001
        logger.info("standing image-edit /faces failed (%s); falling back", e)
        return None
    return out if isinstance(out, dict) and isinstance(out.get("faces"), list) else None


queue = FullEditQueue()
