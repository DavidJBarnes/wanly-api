"""Launch and terminate RunPod GPU workers from the console.

The RunPod key stays server-side. Creating a pod requires a read/write key — one that can also
terminate pods and delete volumes — and there is no launch-only scope, so it must never be
handed to the browser.
"""

import re

from fastapi import APIRouter, Depends, HTTPException

from app import runpod_client
from app.auth import get_current_user
from app.config import settings
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
        raise HTTPException(
            status_code=503,
            detail=(
                f"No {settings.runpod_gpu_type_id} capacity in "
                f"{settings.runpod_datacenter_id} right now. The datacenter is pinned because "
                f"the network volume holding the models is region-locked to it. Try again "
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
