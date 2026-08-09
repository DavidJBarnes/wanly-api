"""Is there work with nobody to do it?

The 3090's host rebooted on 2026-08-08 and its container had restart=no, so the worker never came
back. It was found thirteen hours later, by hand, with four segments queued the whole time. Every
fact needed to catch it was already in the database -- a stale last_heartbeat, a worker marked
offline, segments sitting pending. Nothing put them together.

The distinction that matters, and the one the UI did not draw: an offline worker with an empty
queue is housekeeping, while an offline worker with queued work is an outage. They looked
identical.

The judgement lives here as a pure function for the same reason reservations.decide() does: the
interesting cases are combinatorial rather than temporal, and they are worth pinning as a table
instead of through a database and a page render.
"""

from dataclasses import dataclass
from datetime import datetime

# Statuses that mean a worker can pick up work. "draining" deliberately does NOT count: a draining
# worker finishes what it holds and takes nothing new, so a queue with only draining workers is
# every bit as stalled as one with none -- it just has not noticed yet.
LIVE_STATUSES = frozenset({"online-idle", "online-busy"})


@dataclass(frozen=True)
class QueueHealth:
    pending_segments: int
    live_workers: int
    stalled: bool
    last_worker_seen: datetime | None = None

    @property
    def summary(self) -> str:
        if not self.stalled:
            return ""
        s = "s" if self.pending_segments != 1 else ""
        return f"{self.pending_segments} segment{s} queued and no workers online"


def assess(
    *,
    pending_segments: int,
    worker_statuses: list[str],
    last_worker_seen: datetime | None = None,
) -> QueueHealth:
    """Stalled means there is work AND nobody who can take it.

    Both halves are required. Queued work with a busy worker is a queue doing its job, and no
    workers with an empty queue is a quiet night -- neither is worth interrupting anyone over.
    Raising on either alone would produce an alarm that fires constantly and then gets ignored,
    which is worse than the silence it replaced.
    """
    live = sum(1 for s in worker_statuses if s in LIVE_STATUSES)
    return QueueHealth(
        pending_segments=pending_segments,
        live_workers=live,
        stalled=pending_segments > 0 and live == 0,
        last_worker_seen=last_worker_seen,
    )
