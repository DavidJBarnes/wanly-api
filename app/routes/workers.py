import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import verify_api_key, verify_api_key_or_bearer
from app.database import get_db
from app.enums import JobStatus, SegmentStatus
from app.models import Job, Segment, Worker
from app.queue_health import assess
from app.schemas.workers import QueueHealthResponse, WorkerDrain, WorkerHeartbeat, WorkerRegister, WorkerRename, WorkerResponse, WorkerStatusUpdate

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
        # Re-registering after a container restart can land on a new pod id.
        if body.runpod_pod_id:
            worker.runpod_pod_id = body.runpod_pod_id
        worker.status, worker.drain_after_jobs = reregistered_drain_state(
            worker.status, worker.drain_after_jobs
        )
        worker.last_heartbeat = datetime.now(timezone.utc)
    else:
        worker = Worker(
            friendly_name=body.friendly_name,
            hostname=body.hostname,
            ip_address=body.ip_address,
            comfyui_running=body.comfyui_running,
            runpod_pod_id=body.runpod_pod_id,
        )
        db.add(worker)

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

    rows = (await db.execute(select(Worker.status, Worker.last_heartbeat))).all()
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
