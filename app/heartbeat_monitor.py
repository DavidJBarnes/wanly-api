import asyncio
import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import update

from app.config import settings
from app.database import async_session
from app.models import Worker

logger = logging.getLogger(__name__)


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
                    .where(Worker.last_heartbeat < cutoff, Worker.status != "offline")
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
