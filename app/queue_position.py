"""Where a newly queued segment lands in the queue (wanly-api#471).

Workers claim by job priority, then by segment age. A job's priority is set once, when it is
created, so a continuation, re-roll or retry has always kept its job's slot. That is "top": a
job already under way finishes ahead of jobs queued after it.

"bottom" moves the job behind every other job each time it queues new work. The other jobs
then go first, so the queue rotates across jobs instead of one chain running to the end
while everything else waits.

Only the user-facing queue moves. Regularization renders sit past REG_PRIORITY_BASE so that
real work never waits behind them; a job sent to the back lands ahead of them, the same as a
new job does.
"""
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import AppSetting, Job
from app.regularization import REG_PRIORITY_BASE

SETTING_KEY = "segment_queue_position"
TOP = "top"
BOTTOM = "bottom"
DEFAULT = TOP


async def queue_position(db: AsyncSession) -> str:
    row = await db.get(AppSetting, SETTING_KEY)
    return BOTTOM if row is not None and row.value == BOTTOM else DEFAULT


async def place_queued_job(db: AsyncSession, job: Job) -> None:
    """Call wherever a segment is queued as new work. A no-op unless the setting is "bottom".

    Stale-claim recovery does NOT call this: a segment handed back after a dead worker is the
    same work resuming, and should not lose its place because a machine fell over.
    """
    if job.priority >= REG_PRIORITY_BASE or await queue_position(db) != BOTTOM:
        return
    last = (await db.execute(
        select(func.max(Job.priority)).where(
            Job.user_id == job.user_id, Job.id != job.id, Job.priority < REG_PRIORITY_BASE)
    )).scalar_one()
    if last is not None and last >= job.priority:
        job.priority = last + 1
