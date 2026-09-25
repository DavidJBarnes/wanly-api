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

    Returning a ticket instead (202 + poll) is the next step, and it is what removes the
    timeout cliff entirely rather than widening it. It needs the console to poll, so it is
    not free, and it is not in here yet.

asyncio.Lock hands the lock to waiters in the order they arrived, which is the ordering this
depends on.
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class CaptionQueue:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        #: Arrival order of the paths still waiting. Not a set: position is the point.
        self._waiting: list[str] = []
        self._running: str | None = None

    @asynccontextmanager
    async def turn(self, path: str):
        """Wait for the captioner, then hold it for the duration of the block."""
        self._waiting.append(path)
        try:
            async with self._lock:
                try:
                    self._waiting.remove(path)
                except ValueError:
                    pass
                self._running = path
                try:
                    yield
                finally:
                    self._running = None
        finally:
            # Belt and braces: a cancellation between append and acquire must not leave a
            # phantom in the line, inflating every position reported after it.
            try:
                self._waiting.remove(path)
            except ValueError:
                pass

    def status(self, path: str) -> dict:
        """Where this path sits, computed on read.

        Never stored: a position recorded a moment ago is wrong as soon as anything ahead
        of it finishes.
        """
        if self._running == path:
            return {"status": "running", "position": 0, "depth": self.depth()}
        if path in self._waiting:
            return {"status": "queued",
                    "position": self._waiting.index(path) + 1,
                    "depth": self.depth()}
        return {"status": None, "position": None, "depth": self.depth()}

    def depth(self) -> int:
        """Everything not yet finished, including the one in progress."""
        return len(self._waiting) + (1 if self._running is not None else 0)

    def running_path(self) -> str | None:
        """The image being captioned right now, if any."""
        return self._running

    def waiting_paths(self) -> list[str]:
        """Everything still in line, in the order it will be taken."""
        return list(self._waiting)


#: One queue per process, because there is one captioner.
queue = CaptionQueue()
