"""Launch and terminate RunPod GPU workers from the console.

The RunPod key stays server-side. Creating a pod requires a read/write key — one that can also
terminate pods and delete volumes — and there is no launch-only scope, so it must never be
handed to the browser.
"""

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
    RunPodLaunchRequest,
    RunPodWorker,
)

router = APIRouter()

# RunPod pod names, and the FRIENDLY_NAME the worker registers under, both end up in logs and
# on the Workers page. Keep them boring.
_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9 ._-]{0,48}$")


@router.get("/runpod/availability", response_model=RunPodAvailability,
            dependencies=[Depends(get_current_user)])
async def runpod_availability():
    """Is the configured GPU purchasable in the configured datacenter right now?"""
    try:
        return await runpod_client.get_availability()
    except RunPodError as e:
        raise HTTPException(status_code=503, detail=str(e))


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

    # Check stock first so "no 4090s right now" reads as exactly that, instead of surfacing as
    # a RunPod spec error. This is best-effort — availability flaps, so the launch below can
    # still fail — but it turns the common case into a useful message.
    try:
        availability = await runpod_client.get_availability()
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
                f"No {settings.runpod_gpu_type_id} capacity {where} "
                f"({settings.runpod_cloud_type.lower()} cloud) right now.{why} Try again "
                f"shortly — availability changes minute to minute."
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
        return await runpod_client.launch_worker(name, env)
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
