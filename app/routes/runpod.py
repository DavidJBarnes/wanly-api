"""Launch and terminate RunPod GPU workers from the console.

The RunPod key stays server-side. Creating a pod requires a read/write key — one that can also
terminate pods and delete volumes — and there is no launch-only scope, so it must never be
handed to the browser.
"""

import asyncio
import re
import uuid
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app import runpod_client
from app.auth import get_current_user
from app.config import settings
from app.database import get_db
from app.models import GpuReservation
from app.reservations import ReservationStatus
from app.schemas.reservations import ReservationCreate, ReservationResponse
from app.runpod_client import RunPodError
from app.schemas.runpod import (
    RunPodAvailability,
    RunPodGpuOption,
    RunPodLaunchRequest,
    RunPodWorker,
)

router = APIRouter()

# RunPod pod names, and the FRIENDLY_NAME the worker registers under, both end up in logs and
# on the Workers page. Keep them boring.
_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9 ._-]{0,48}$")


@router.get("/runpod/availability", response_model=RunPodAvailability,
            dependencies=[Depends(get_current_user)])
async def runpod_availability(gpu_type_id: str | None = None):
    """Is this GPU priced in the configured datacenter right now?

    Note this is not the same question as "can a pod be placed" — see get_availability().
    """
    if gpu_type_id:
        _validate_gpu(gpu_type_id)
    try:
        return await runpod_client.get_availability(gpu_type_id)
    except RunPodError as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.get("/runpod/gpu-options", response_model=list[RunPodGpuOption],
            dependencies=[Depends(get_current_user)])
async def runpod_gpu_options():
    """The GPUs the launcher offers, each with live price and stock.

    Priced concurrently: one round trip each, and with only a handful of options the serial
    version would visibly stall the dialog for no reason. A lookup that fails is reported as an
    option with an error rather than dropped — an absent 4090 would read as "not supported",
    which is a different and wrong message.
    """
    gpus = runpod_client.selectable_gpus()
    results = await asyncio.gather(
        *(runpod_client.get_availability(g) for g in gpus), return_exceptions=True
    )
    options = []
    for gpu, result in zip(gpus, results):
        if isinstance(result, Exception):
            options.append({
                "gpu_type_id": gpu,
                "is_default": gpu == settings.runpod_gpu_type_id,
                "available": False,
                "error": str(result),
            })
        else:
            options.append({**result, "is_default": gpu == settings.runpod_gpu_type_id})
    return options


def _validate_gpu(gpu_type_id: str) -> str:
    """Reject any GPU not on the server's allowlist.

    The 5090 is the reason this is not a pass-through: it is purchasable on community and our
    workflow does not run on it, so an unvalidated field would let the browser buy a pod that
    cannot do the work.
    """
    allowed = runpod_client.selectable_gpus()
    if gpu_type_id not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported GPU {gpu_type_id!r}. Choose one of: {', '.join(allowed)}.",
        )
    return gpu_type_id


@router.get("/runpod/workers", response_model=list[RunPodWorker],
            dependencies=[Depends(get_current_user)])
async def list_runpod_workers():
    try:
        return await runpod_client.list_workers()
    except RunPodError as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.post("/runpod/workers", response_model=RunPodWorker, status_code=201,
             dependencies=[Depends(get_current_user)])
async def launch_runpod_worker(body: RunPodLaunchRequest):
    """Launch a worker, refusing early and legibly when there is no capacity."""
    name = (body.name or "").strip()
    if not _NAME_RE.match(name):
        raise HTTPException(
            status_code=400,
            detail="Name must be 1-49 chars: letters, numbers, spaces, dot, dash, underscore.",
        )

    gpu = _validate_gpu(body.gpu_type_id) if body.gpu_type_id else settings.runpod_gpu_type_id

    # Check price first so a GPU RunPod does not sell here fails fast and legibly.
    #
    # Only a NEGATIVE result blocks. A positive one is not permission to launch: community 4090
    # reported available=True / stock="Low" for an hour while every create failed, because
    # lowestPrice answers "is this GPU sold here", not "will a host take this pod". Treating the
    # positive as a green light was harmless; what hurt was having no path past it, so the
    # placement failure never got a chance to produce its own, far more specific message.
    try:
        availability = await runpod_client.get_availability(gpu)
    except RunPodError as e:
        raise HTTPException(status_code=503, detail=str(e))

    if not availability["available"]:
        where = (
            f"in {settings.runpod_datacenter_id}"
            if settings.runpod_datacenter_id
            else "on any datacenter"
        )
        why = (
            " The datacenter is pinned because the network volume holding the models is "
            "region-locked to it."
            if settings.runpod_datacenter_id
            else ""
        )
        raise HTTPException(
            status_code=503,
            detail=(
                f"RunPod does not currently price {gpu} {where} "
                f"({settings.runpod_cloud_type.lower()} cloud).{why} Try again "
                f"shortly, or pick a different GPU — availability changes minute to minute."
            ),
        )

    # The worker needs the queue to claim from, and its own name so it registers legibly rather
    # than as runpod-<podid>. QUEUE_API_KEY is this server's own daemon key.
    env = {
        "FRIENDLY_NAME": name,
        "QUEUE_URL": body.queue_url or settings.runpod_worker_queue_url,
        "QUEUE_API_KEY": settings.api_key,
    }
    if settings.runpod_api_key:
        # Lets the worker stop its own pod when drained. Without it a drain leaves the pod
        # running and the container simply respawns.
        env["RUNPOD_API_KEY"] = settings.runpod_api_key

    try:
        return await runpod_client.launch_worker(name, env, gpu)
    except RunPodError as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.delete("/runpod/workers/{pod_id}", status_code=204,
               dependencies=[Depends(get_current_user)])
async def terminate_runpod_worker(pod_id: str):
    """Terminate a pod outright.

    This is destructive and immediate: it does not wait for in-flight work. Draining from the
    Workers page is the graceful path. Terminate exists for when the user has decided the pod
    should go now.
    """
    try:
        await runpod_client.terminate_worker(pod_id)
    except RunPodError as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.get("/runpod/reservations", response_model=list[ReservationResponse],
            dependencies=[Depends(get_current_user)])
async def list_reservations(db: AsyncSession = Depends(get_db)):
    """Reservations that are still waiting, newest first.

    Terminal ones are not returned: the point of this list is "what is still going to spend
    money", and a wall of expired rows buries that.
    """
    rows = (
        await db.execute(
            select(GpuReservation)
            .where(GpuReservation.status == ReservationStatus.PENDING)
            .order_by(GpuReservation.created_at.desc())
        )
    ).scalars().all()
    return rows


@router.post("/runpod/reservations", response_model=ReservationResponse, status_code=201,
             dependencies=[Depends(get_current_user)])
async def create_reservation(body: ReservationCreate, db: AsyncSession = Depends(get_db)):
    """Keep trying to launch a worker until one is available or the window closes."""
    name = (body.name or "").strip()
    if not _NAME_RE.match(name):
        raise HTTPException(
            status_code=400,
            detail="Name must be 1-49 chars: letters, numbers, spaces, dot, dash, underscore.",
        )

    # One pending reservation per name, or two of them race to launch pods that would both try
    # to register as the same worker.
    existing = (
        await db.execute(
            select(GpuReservation).where(
                GpuReservation.name == name,
                GpuReservation.status == ReservationStatus.PENDING,
            )
        )
    ).scalar_one_or_none()
    if existing:
        raise HTTPException(
            status_code=409,
            detail=f"A reservation named '{name}' is already waiting.",
        )

    reservation = GpuReservation(
        id=uuid.uuid4(),
        name=name,
        status=ReservationStatus.PENDING,
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=body.minutes),
        drain_after_jobs=body.drain_after_jobs,
        gpu_type_id=_validate_gpu(body.gpu_type_id) if body.gpu_type_id else None,
    )
    db.add(reservation)
    await db.commit()
    await db.refresh(reservation)
    return reservation


@router.delete("/runpod/reservations/{reservation_id}", status_code=204,
               dependencies=[Depends(get_current_user)])
async def cancel_reservation(reservation_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """Cancel a pending reservation.

    Does not touch a pod that has already launched — by then it is a worker, and the Workers
    page is where a worker is stopped.
    """
    reservation = await db.get(GpuReservation, reservation_id)
    if not reservation:
        raise HTTPException(status_code=404, detail="Reservation not found")
    if reservation.status == ReservationStatus.PENDING:
        reservation.status = ReservationStatus.CANCELLED
        await db.commit()
