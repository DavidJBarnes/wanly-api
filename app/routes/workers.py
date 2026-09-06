import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import verify_api_key, verify_api_key_or_bearer
from app.database import get_db
from app.enums import JobStatus, SegmentStatus, WorkerKind, WorkerStatus
from app.models import Job, Segment, Worker
from app.queue_health import COUNTED_KINDS, assess
from app.schemas.workers import QueueHealthResponse, WorkerDrain, WorkerHeartbeat, WorkerRegister, WorkerRename, WorkerResponse, WorkerStatusUpdate

logger = logging.getLogger(__name__)

router = APIRouter()


def reregistered_drain_state(
    current_status: str | None, current_drain_after: int | None
) -> tuple[str, int | None]:
    """Status + drain countdown for a worker that is re-registering.

    A re-register must NOT cancel a pending drain. The daemon re-registers on every
    start, and a RunPod container respawns automatically when the daemon exits — so
    resetting here silently erased operator drain requests roughly 30s after they took
    effect, and the worker went straight back to claiming work. Cancelling a drain is
    an explicit action: DELETE /workers/{id}/drain.
    """
    if current_status == "draining":
        return "draining", None
    return "online-idle", current_drain_after


@router.post("/workers", response_model=WorkerResponse, status_code=201, dependencies=[Depends(verify_api_key)])
async def register_worker(body: WorkerRegister, db: AsyncSession = Depends(get_db)):
    # Upsert: if friendly_name already exists, reclaim that row
    result = await db.execute(
        select(Worker).where(Worker.friendly_name == body.friendly_name)
    )
    worker = result.scalar_one_or_none()
    if worker:
        worker.hostname = body.hostname
        worker.ip_address = body.ip_address
        worker.comfyui_running = body.comfyui_running
        # Re-registering can legitimately change what a box is: the same host could stop
        # running services and start running an engine. Taken from the request rather than
        # preserved, so the row follows reality instead of the first thing it ever saw.
        worker.kind = body.kind
        if body.provides is not None:
            worker.provides = body.provides
        # Re-registering after a container restart can land on a new pod id.
        if body.runpod_pod_id:
            worker.runpod_pod_id = body.runpod_pod_id
        if worker.kind == WorkerKind.RENDER:
            worker.status, worker.drain_after_jobs = reregistered_drain_state(
                worker.status, worker.drain_after_jobs
            )
        else:
            # A service has nothing to drain, and must not spend its first 30 seconds saying
            # "idle" -- the word #269 exists to stop applying to it. Without this the row is
            # created on the model's default, online-idle, and only corrects on the first
            # heartbeat; the Workers page renders "Idle" on a service for that whole window,
            # which is exactly the ambiguity the separate vocabulary was introduced to end.
            worker.status = WorkerStatus.ONLINE
        worker.last_heartbeat = datetime.now(timezone.utc)
    else:
        worker = Worker(
            friendly_name=body.friendly_name,
            hostname=body.hostname,
            ip_address=body.ip_address,
            comfyui_running=body.comfyui_running,
            runpod_pod_id=body.runpod_pod_id,
            kind=body.kind,
            provides=body.provides,
            # Same reason as above: the column default is online-idle, which is a render word.
            status=(WorkerStatus.ONLINE_IDLE if body.kind == WorkerKind.RENDER
                    else WorkerStatus.ONLINE),
        )
        db.add(worker)

    # A worker that is registering holds nothing. Release anything still assigned to it.
    await _release_orphaned_claims(db, worker)

    # A reservation may have asked for a drain policy up front. Apply it the moment the worker
    # it was waiting for appears.
    #
    # It has to happen here rather than at launch: the pod exists minutes before the worker
    # registers, and drain_after_jobs lives on the worker row, which does not exist until now.
    # Without this a reservation made at 11pm produces a worker that runs until someone notices
    # — which is the whole thing the option was added to prevent.
    await _apply_reserved_drain(db, worker)

    await db.commit()
    await db.refresh(worker)
    return worker


async def _release_orphaned_claims(db: AsyncSession, worker: Worker) -> None:
    """Free any segment still assigned to a worker that has just registered.

    Registration means the daemon has started fresh: it happens once, before the claim loop,
    so a worker reaching this point is by definition rendering nothing. Any segment still
    marked CLAIMED or PROCESSING against it belongs to a previous life of that worker, and
    nothing will ever finish it.

    WHY NOTHING ELSE CATCHES THIS
        The reclaim in /segments/next needs either a stale heartbeat or an idle worker with
        an empty progress log. A replaced container satisfies neither: registration upserts
        on friendly_name, so the row -- and its id -- is REUSED, the new daemon heartbeats
        immediately, and it goes on to claim something else and report online-busy. The old
        claim is pinned to a live, healthy, busy worker and is unreachable by every existing
        rule.

        Observed 2026-09-06: a container was replaced 50% through a 673 MB LoRA download in
        [2/6]. The segment sat in PROCESSING for SEVEN HOURS against a worker that had never
        heard of it, and its job could not finish. It had to be freed by hand.

    A worker renders one segment at a time -- the claim loop guards it with executing_event --
    so there is no case where a registering worker legitimately holds work. Deliberately not
    tied to container replacement specifically: any path that gets a daemon back to
    registration (crash, OOM kill, manual restart, a pod recreated by the updater) leaves the
    same wreckage, and they should all be cleaned up by the same rule.

    Progress log is cleared with the claim. It describes an attempt that no longer exists, and
    leaving it would also make the segment permanently ineligible for the live-worker reclaim
    in /segments/next, which requires an empty log.
    """
    if worker.id is None:
        return  # brand new row; it cannot hold anything yet
    result = await db.execute(
        update(Segment)
        .where(
            Segment.worker_id == worker.id,
            Segment.status.in_([SegmentStatus.CLAIMED, SegmentStatus.PROCESSING]),
        )
        .values(status=SegmentStatus.PENDING, worker_id=None, worker_name=None,
                claimed_at=None, progress_log=None)
        .returning(Segment.id)
    )
    freed = result.scalars().all()
    if freed:
        logger.warning(
            "Worker %s re-registered holding %d claimed segment(s); released to PENDING: %s",
            worker.friendly_name, len(freed), ", ".join(str(i) for i in freed),
        )


async def _apply_reserved_drain(db: AsyncSession, worker: Worker) -> None:
    from app.models import GpuReservation

    result = await db.execute(
        select(GpuReservation).where(
            GpuReservation.name == worker.friendly_name,
            GpuReservation.drain_after_jobs.isnot(None),
            GpuReservation.status == "launched",
        )
    )
    reservation = result.scalars().first()
    # Only set it on a fresh registration. Overwriting an existing countdown would restart it
    # every time the worker re-registers, which on a container restart means it never drains.
    if reservation and worker.drain_after_jobs is None and worker.status != "draining":
        worker.drain_after_jobs = reservation.drain_after_jobs


@router.delete("/workers/{worker_id}", status_code=204, dependencies=[Depends(verify_api_key_or_bearer)])
async def deregister_worker(
    worker_id: uuid.UUID, db: AsyncSession = Depends(get_db)
):
    worker = await db.get(Worker, worker_id)
    if not worker:
        raise HTTPException(status_code=404, detail="Worker not found")
    await db.delete(worker)
    await db.commit()


@router.post("/workers/{worker_id}/drain", response_model=WorkerResponse, dependencies=[Depends(verify_api_key_or_bearer)])
async def drain_worker(
    worker_id: uuid.UUID,
    body: WorkerDrain | None = None,
    db: AsyncSession = Depends(get_db),
):
    worker = await db.get(Worker, worker_id)
    if not worker:
        raise HTTPException(status_code=404, detail="Worker not found")
    if worker.status == "offline":
        raise HTTPException(status_code=400, detail="Cannot drain an offline worker")
    after_jobs = body.after_jobs if body else None
    if after_jobs and after_jobs > 0:
        worker.drain_after_jobs = after_jobs
    else:
        worker.status = "draining"
        worker.drain_after_jobs = None
    await db.commit()
    await db.refresh(worker)
    return worker


@router.delete("/workers/{worker_id}/drain", response_model=WorkerResponse, dependencies=[Depends(verify_api_key_or_bearer)])
async def cancel_drain(
    worker_id: uuid.UUID, db: AsyncSession = Depends(get_db)
):
    worker = await db.get(Worker, worker_id)
    if not worker:
        raise HTTPException(status_code=404, detail="Worker not found")
    worker.drain_after_jobs = None
    if worker.status == "draining":
        worker.status = "online-idle"
    await db.commit()
    await db.refresh(worker)
    return worker


@router.post("/workers/{worker_id}/heartbeat", response_model=WorkerResponse, dependencies=[Depends(verify_api_key)])
async def heartbeat(
    worker_id: uuid.UUID,
    body: WorkerHeartbeat,
    db: AsyncSession = Depends(get_db),
):
    worker = await db.get(Worker, worker_id)
    if not worker:
        raise HTTPException(status_code=404, detail="Worker not found")
    worker.last_heartbeat = datetime.now(timezone.utc)
    worker.comfyui_running = body.comfyui_running
    if body.gpu_stats is not None:
        worker.gpu_stats = body.gpu_stats
    worker.sd_scripts = body.sd_scripts
    worker.a1111 = body.a1111
    # `is not None`, unlike a1111 above: an older daemon omits this entirely, and writing
    # None would erase a good inventory every heartbeat during a rolling upgrade.
    if body.loras is not None:
        worker.loras = body.loras
    # Same conditional write, same reason: an older daemon omits the field on every
    # heartbeat, and assigning None would blank a good list seconds after a newer worker
    # reported it.
    if body.checkpoints is not None:
        worker.checkpoints = body.checkpoints
    if body.fetchable_kinds is not None:
        worker.fetchable_kinds = body.fetchable_kinds
    # Same conditional write again. These two answer "what code is this worker running"
    # (wanly-gpu-docker#72) and they move independently: the daemon is re-cloned from main at
    # every container boot, the image only changes on pull + recreate.
    if body.daemon_commit is not None:
        worker.daemon_commit = body.daemon_commit
    if body.image_ref is not None:
        worker.image_ref = body.image_ref
    # Same conditional write once more. A services container can change what it runs across a
    # restart without re-registering, so the row would otherwise advertise yesterday's set.
    if body.provides is not None:
        worker.provides = body.provides

    if worker.kind != WorkerKind.RENDER:
        # A SERVICE REPORTS ITS OWN HEALTH, because there is no claim state to derive one
        # from -- it is never idle-waiting-for-work and never busy-with-a-segment. It sends
        # "online" when everything it was asked to run answers and "degraded" when some of it
        # does not, which is a distinction the render vocabulary cannot express and which
        # wanly-gpu-docker#80 is open because we could not express.
        #
        # `status` is honoured ONLY here. A render worker's status stays derived, so a daemon
        # cannot talk itself into looking idle.
        if body.status is not None:
            worker.status = body.status
        elif worker.status == "offline":
            worker.status = WorkerStatus.ONLINE
    else:
        if worker.status == "offline":
            worker.status = "online-idle"
        # If sd-scripts is actively training, worker can't be idle
        if worker.status not in ("offline", "draining"):
            sd_training = (
                body.sd_scripts.get("sd_scripts_training", False)
                if body.sd_scripts
                else False
            )
            if sd_training:
                worker.status = "online-busy"
    await db.commit()
    await db.refresh(worker)
    return worker


@router.patch("/workers/{worker_id}/friendly_name", response_model=WorkerResponse, dependencies=[Depends(verify_api_key_or_bearer)])
async def rename_worker(
    worker_id: uuid.UUID,
    body: WorkerRename,
    db: AsyncSession = Depends(get_db),
):
    worker = await db.get(Worker, worker_id)
    if not worker:
        raise HTTPException(status_code=404, detail="Worker not found")
    worker.friendly_name = body.friendly_name.strip()
    await db.commit()
    await db.refresh(worker)
    return worker


@router.patch("/workers/{worker_id}/status", response_model=WorkerResponse, dependencies=[Depends(verify_api_key)])
async def update_status(
    worker_id: uuid.UUID,
    body: WorkerStatusUpdate,
    db: AsyncSession = Depends(get_db),
):
    allowed = {"online-idle", "online-busy"}
    if body.status not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Status must be one of: {', '.join(sorted(allowed))}",
        )
    worker = await db.get(Worker, worker_id)
    if not worker:
        raise HTTPException(status_code=404, detail="Worker not found")
    if worker.status == "draining":
        return worker
    worker.status = body.status
    if body.status == "online-idle" and worker.drain_after_jobs is not None:
        worker.drain_after_jobs -= 1
        if worker.drain_after_jobs <= 0:
            worker.status = "draining"
            worker.drain_after_jobs = None
    await db.commit()
    await db.refresh(worker)
    return worker


@router.get("/workers", response_model=list[WorkerResponse], dependencies=[Depends(verify_api_key_or_bearer)])
async def list_workers(
    status: str | None = Query(default=None),
    db: AsyncSession = Depends(get_db),
):
    stmt = select(Worker)
    if status:
        stmt = stmt.where(Worker.status == status)
    result = await db.execute(stmt)
    return result.scalars().all()


@router.get("/queue-health", response_model=QueueHealthResponse,
            dependencies=[Depends(verify_api_key_or_bearer)])
async def queue_health(db: AsyncSession = Depends(get_db)):
    """Is there queued work with nobody to do it?

    Cheap enough to poll from any page. Counts only segments whose JOB is also live -- an
    archived job's segments are unclaimable by design and would otherwise raise a permanent
    false alarm (see #177).
    """
    pending = (await db.execute(
        select(func.count())
        .select_from(Segment)
        .join(Job, Segment.job_id == Job.id)
        .where(
            Segment.status == SegmentStatus.PENDING,
            Job.status.in_([JobStatus.PENDING, JobStatus.PROCESSING]),
        )
    )).scalar() or 0

    # Render workers only. A service can never take a segment, so letting one count as live
    # would silence `stalled` permanently -- see COUNTED_KINDS in app/queue_health.py.
    rows = (await db.execute(
        select(Worker.status, Worker.last_heartbeat)
        .where(Worker.kind.in_(COUNTED_KINDS))
    )).all()
    health = assess(
        pending_segments=pending,
        worker_statuses=[r[0] for r in rows],
        last_worker_seen=max((r[1] for r in rows), default=None),
    )
    return QueueHealthResponse(
        pending_segments=health.pending_segments,
        live_workers=health.live_workers,
        stalled=health.stalled,
        last_worker_seen=health.last_worker_seen,
        summary=health.summary,
    )


@router.get("/workers/{worker_id}", response_model=WorkerResponse, dependencies=[Depends(verify_api_key_or_bearer)])
async def get_worker(
    worker_id: uuid.UUID, db: AsyncSession = Depends(get_db)
):
    worker = await db.get(Worker, worker_id)
    if not worker:
        raise HTTPException(status_code=404, detail="Worker not found")
    return worker
