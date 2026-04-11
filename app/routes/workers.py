import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import verify_api_key, verify_api_key_or_bearer
from app.database import get_db
from app.models import Worker
from app.schemas.workers import WorkerDrain, WorkerHeartbeat, WorkerRegister, WorkerRename, WorkerResponse, WorkerStatusUpdate

router = APIRouter()


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
        worker.status = "online-idle"
        worker.drain_after_jobs = None
        worker.last_heartbeat = datetime.now(timezone.utc)
    else:
        worker = Worker(
            friendly_name=body.friendly_name,
            hostname=body.hostname,
            ip_address=body.ip_address,
            comfyui_running=body.comfyui_running,
        )
        db.add(worker)
    await db.commit()
    await db.refresh(worker)
    return worker


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


@router.get("/workers/{worker_id}", response_model=WorkerResponse, dependencies=[Depends(verify_api_key_or_bearer)])
async def get_worker(
    worker_id: uuid.UUID, db: AsyncSession = Depends(get_db)
):
    worker = await db.get(Worker, worker_id)
    if not worker:
        raise HTTPException(status_code=404, detail="Worker not found")
    return worker
