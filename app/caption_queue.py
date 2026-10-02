"""One caption at a time, with a queue the caller can see.

WHY THIS EXISTS

    The captioner is a single-slot resource: ollama runs OLLAMA_NUM_PARALLEL=1, so requests
    never overlap no matter how many arrive. Firing N describes at once therefore does not
    make them finish sooner -- it makes each one wait for all the others INSIDE an HTTP
    request. Measured on the 3090 at ~25s a call, two calls an image:

        18s  28s  18s  27s  22s  21s  37s  48s  51s  1m10  55s  1m17  ...

    and at IMAGE_DESCRIPTION_TIMEOUT_S the back of the queue starts failing -- with the
    front half of the same image already written, which is exactly how "scene described,
    but the motion caption failed" happens. Raising the timeout only moves the cliff.

    The queue does two things the bare race could not: the work happens in a KNOWN ORDER,
    and every response can say where it sits, so the UI can show "3rd of 7" instead of a
    spinner indistinguishable from a hang.

WHY A TURNSTILE AND NOT A BACKGROUND WORKER

    The caller still gets its words back. RecipeForm describes a start frame and uses the
    reply (console#427), so the request has to carry the result. Taking a turn keeps that
    contract intact -- the work still runs on the request's own session -- while making the
    waiting orderly.

    Returning a ticket instead (202 + poll) is what removes the timeout cliff entirely, and
    since console#564 that is what describe does: app/caption_tickets.py takes the turn in a
    background task and the console polls. The turn itself is unchanged.

asyncio.Lock hands the lock to waiters in the order they arrived, which is the ordering this
depends on.
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class _Entry:
    """One place in line: the image, who asked (kind), and an optional token that identifies
    this particular request -- a caption ticket's id -- so its own position can be read even
    when the same image is in line twice."""
    __slots__ = ("path", "kind", "token")

    def __init__(self, path: str, kind: str, token: str | None) -> None:
        self.path = path
        self.kind = kind
        self.token = token


class CaptionQueue:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        #: Arrival order of the entries still waiting. Not a set: position is the point.
        self._waiting: list[_Entry] = []
        self._running: _Entry | None = None

    def reserve(self, path: str, kind: str = "describe", token: str | None = None) -> _Entry:
        """Take a place in line NOW, synchronously, for a turn() that starts a moment later.

        A caption ticket answers its HTTP request before its task has run a single step; the
        ticket it returns must already have a position, or the first poll says "not queued".
        Pass the entry to turn(reserved=...). Discard it if the turn never happens.
        """
        entry = _Entry(path, kind, token)
        self._waiting.append(entry)
        return entry

    @asynccontextmanager
    async def turn(self, path: str, kind: str = "describe", token: str | None = None,
                   reserved: _Entry | None = None):
        """Wait for the captioner, then hold it for the duration of the block.

        `kind` says who is asking -- "describe" (a caption ticket for the modal, the grid, a
        dialog), "hold" (a held job), "dataset" (a training caption) or "try" (a Settings
        preview that stores nothing) -- so a view can tell a caption that will land on the
        image from one that will not.
        """
        if reserved is not None:
            entry = reserved
        else:
            entry = _Entry(path, kind, token)
            self._waiting.append(entry)
        try:
            async with self._lock:
                self.discard(entry)
                self._running = entry
                try:
                    yield
                finally:
                    self._running = None
        finally:
            # Belt and braces: a cancellation between append and acquire must not leave a
            # phantom in the line, inflating every position reported after it.
            self.discard(entry)

    def discard(self, entry: _Entry) -> None:
        # By identity: the same image can be in line twice (a dataset caption and a describe).
        for i, e in enumerate(self._waiting):
            if e is entry:
                del self._waiting[i]
                return

    def status(self, path: str) -> dict:
        """Where this path sits, computed on read.

        Never stored: a position recorded a moment ago is wrong as soon as anything ahead
        of it finishes.
        """
        if self._running is not None and self._running.path == path:
            return {"status": "running", "position": 0, "depth": self.depth()}
        for i, e in enumerate(self._waiting):
            if e.path == path:
                return {"status": "queued", "position": i + 1, "depth": self.depth()}
        return {"status": None, "position": None, "depth": self.depth()}

    def token_status(self, token: str) -> dict:
        """status(), for one request rather than for an image: a caption ticket's own place."""
        if self._running is not None and self._running.token == token:
            return {"status": "running", "position": 0, "depth": self.depth()}
        for i, e in enumerate(self._waiting):
            if e.token == token:
                return {"status": "queued", "position": i + 1, "depth": self.depth()}
        return {"status": None, "position": None, "depth": self.depth()}

    def in_flight(self, path: str, kinds=None) -> bool:
        """Is a caption of this image queued or running right now, from anywhere?

        The single-flight check the caption hold makes (console#562): a job waiting on this
        image joins the caption the modal, the New Job dialog or another job already started,
        rather than queueing a second one that would overwrite the first with different words.
        `kinds` narrows it to turns of those kinds (a dataset caption saves no scene words).
        """
        def match(e: _Entry) -> bool:
            return e.path == path and (kinds is None or e.kind in kinds)
        if self._running is not None and match(self._running):
            return True
        return any(match(e) for e in self._waiting)

    def depth(self) -> int:
        """Everything not yet finished, including the one in progress."""
        return len(self._waiting) + (1 if self._running is not None else 0)

    def running_path(self) -> str | None:
        """The image being captioned right now, if any."""
        return self._running.path if self._running is not None else None

    def waiting_paths(self) -> list[str]:
        """Everything still in line, in the order it will be taken."""
        return [e.path for e in self._waiting]

    def entries(self) -> list[dict]:
        """The whole line, running first: {path, kind, token, status, position}.

        What lets one poll annotate every image on a page (console#564) instead of one
        request per image.
        """
        out = []
        if self._running is not None:
            r = self._running
            out.append({"path": r.path, "kind": r.kind, "token": r.token,
                        "status": "running", "position": 0})
        for i, e in enumerate(self._waiting):
            out.append({"path": e.path, "kind": e.kind, "token": e.token,
                        "status": "queued", "position": i + 1})
        return out


#: THE SCENE LANE. One queue per captioner, because each is one ollama slot. Named `queue`
#: because it was the only one, and everything that is not a caption ticket's motion half --
#: dataset captions, Settings tries, the scene half of every describe -- still stands in it.
queue = CaptionQueue()

#: THE MOTION LANE (wanly-console#572). The motion paragraph is made on a different, slower
#: captioner (Qwen3-VL 32B against JoyCaption's seconds), so it waits in a line of its own: a
#: ticket takes a turn here only after its scene is saved, and the next image's scene does
#: not wait for this image's motion.
motion_queue = CaptionQueue()

SCENE_LANE = "scene"
MOTION_LANE = "motion"


def lanes() -> list[tuple[str, CaptionQueue]]:
    """Both lanes, scene first. Read at call time: tests replace the module attributes."""
    return [(SCENE_LANE, queue), (MOTION_LANE, motion_queue)]


def total_depth() -> int:
    return sum(q.depth() for _, q in lanes())
