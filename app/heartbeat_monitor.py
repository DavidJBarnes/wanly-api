import asyncio
import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import update

from app.config import settings
from app.database import async_session
from app.models import Worker

logger = logging.getLogger(__name__)


def offline_sweep_conditions(cutoff):
    """Which workers the sweep may mark offline.

    Extracted so the rule has exactly one definition. Inlined in the update statement it could
    only be tested by re-typing it, which is how a test ends up asserting what it wishes were
    true rather than what the code does.
    """
    return (
        # Silent for longer than the threshold.
        Worker.last_heartbeat < cutoff,
        # Already offline: nothing to change.
        Worker.status != "offline",
        # A draining worker keeps its status when it goes quiet.
        #
        # Overwriting it destroyed the drain: heartbeat() sets a previously-offline worker back
        # to online-idle, so it resumed claiming as if nothing had been asked of it. That
        # happened for real — a worker went silent for 240s during identity scoring and came
        # back having forgotten it was draining.
        #
        # The blocking-scoring cause is fixed in wanly-gpu-daemon#117, but a drain must survive
        # silence for ANY reason: network partition, GPU hang, crash. Losing an operator's
        # explicit instruction because a heartbeat was late is never right — an unreachable
        # drained worker should read as draining, not as available.
        Worker.status != "draining",
    )


async def heartbeat_monitor():
    while True:
        await asyncio.sleep(15)
        try:
            cutoff = datetime.now(timezone.utc) - timedelta(
                seconds=settings.heartbeat_offline_seconds
            )
            async with async_session() as session:
                stmt = (
                    update(Worker)
                    .where(*offline_sweep_conditions(cutoff))
                    .values(status="offline", comfyui_running=False)
                    .returning(Worker.id)
                )
                result = await session.execute(stmt)
                marked = result.scalars().all()
                await session.commit()
                if marked:
                    logger.info(
                        "Marked %d worker(s) offline: %s",
                        len(marked),
                        [str(w) for w in marked],
                    )
        except Exception:
            logger.exception("Heartbeat monitor error")
