"""The offline sweep, run against a real database (#162).

tests/test_offline_sweep.py asserts the shape of the WHERE clause. These execute it, which is
the difference between "the predicate mentions draining" and "a draining worker actually
survives the sweep".

That distinction is not academic. The bug in #161 was a missing condition, and a shape test
only catches it because someone thought to assert on that exact string. A behavioural test
catches it because the worker is still draining afterwards.
"""

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select, update

from app.config import settings
from app.heartbeat_monitor import offline_sweep_conditions
from app.models import Worker


async def _worker(db, name: str, status: str, seconds_silent: int) -> Worker:
    w = Worker(
        friendly_name=name,
        hostname="h",
        ip_address="127.0.0.1",
        status=status,
        comfyui_running=True,
        last_heartbeat=datetime.now(timezone.utc) - timedelta(seconds=seconds_silent),
    )
    db.add(w)
    await db.flush()
    return w


async def _run_sweep(db) -> None:
    """Exactly what heartbeat_monitor runs, against this session."""
    cutoff = datetime.now(timezone.utc) - timedelta(
        seconds=settings.heartbeat_offline_seconds
    )
    await db.execute(
        update(Worker)
        .where(*offline_sweep_conditions(cutoff))
        .values(status="offline", comfyui_running=False)
    )
    await db.flush()


async def _status(db, worker_id) -> str:
    return (await db.execute(select(Worker.status).where(Worker.id == worker_id))).scalar_one()


class TestOfflineSweepBehaviour:
    async def test_a_silent_worker_is_marked_offline(self, db):
        w = await _worker(db, "silent", "online-idle", seconds_silent=600)
        await _run_sweep(db)
        assert await _status(db, w.id) == "offline"

    async def test_a_draining_worker_survives_silence(self, db):
        """#161. Marking it offline made heartbeat() reset it to online-idle, so it resumed
        claiming as if the drain had never been requested."""
        w = await _worker(db, "draining-silent", "draining", seconds_silent=600)
        await _run_sweep(db)
        assert await _status(db, w.id) == "draining", (
            "a drain must survive silence for any reason — network partition, GPU hang, crash"
        )

    async def test_a_live_worker_is_untouched(self, db):
        w = await _worker(db, "live", "online-busy", seconds_silent=5)
        await _run_sweep(db)
        assert await _status(db, w.id) == "online-busy"

    async def test_a_busy_but_silent_worker_is_marked_offline(self, db):
        """Not a special case — this is the one that surprised us. A worker mid-segment whose
        event loop was blocked went silent and was correctly swept; the bug was that it took a
        pending drain with it."""
        w = await _worker(db, "busy-silent", "online-busy", seconds_silent=600)
        await _run_sweep(db)
        assert await _status(db, w.id) == "offline"

    @pytest.mark.parametrize("seconds,expected", [
        (settings.heartbeat_offline_seconds - 5, "online-idle"),
        (settings.heartbeat_offline_seconds + 5, "offline"),
    ])
    async def test_the_boundary_is_where_it_claims_to_be(self, db, seconds, expected):
        """A shape test cannot see an off-by-one in the cutoff, or a cutoff built from the
        wrong clock."""
        w = await _worker(db, f"boundary-{seconds}", "online-idle", seconds_silent=seconds)
        await _run_sweep(db)
        assert await _status(db, w.id) == expected

    async def test_rows_do_not_leak_between_tests(self, db):
        """The outer transaction is rolled back, so every test sees only what it created."""
        names = (await db.execute(select(Worker.friendly_name))).scalars().all()
        assert names == [], f"leaked rows from a previous test: {names}"
