"""Thin RunPod client for launching and terminating GPU workers.

Deliberately small. It knows how to ask whether a GPU is available, start one, list what is
running, and terminate it — nothing else. Everything configurable lives in settings so a route
never hardcodes a datacenter or a GPU.

Two RunPod APIs are in play and the split is not arbitrary:

  - REST (rest.runpod.io) for pod create/list/terminate. It is the current API and returns
    ordinary JSON errors.
  - GraphQL (api.runpod.io) for availability, because `gpuTypes(...).lowestPrice` is the only
    place that reports stock for a specific GPU in a specific datacenter, and REST has no
    equivalent.
"""

import httpx

from app.config import settings

REST = "https://rest.runpod.io/v1"
GRAPHQL = "https://api.runpod.io/graphql"

# Long enough for RunPod to actually place a pod, short enough that a hung request does not pin
# a request worker. Pod creation is the slow one; it provisions before responding.
_TIMEOUT = httpx.Timeout(connect=10.0, read=45.0, write=15.0, pool=10.0)


class RunPodError(RuntimeError):
    """A RunPod call failed in a way worth showing the user."""


def _is_secure() -> bool:
    return settings.runpod_cloud_type.upper() == "SECURE"


def selectable_gpus() -> list[str]:
    """The GPUs the launcher may be asked for, in preference order.

    An allowlist rather than "whatever the caller sends": a GPU is only usable if our image runs
    on it, and the 5090 is the standing counter-example. Letting the browser name an arbitrary
    gpuTypeId would make an unsupported pod one typo away.
    """
    names = [g.strip() for g in settings.runpod_gpu_type_ids.split(",") if g.strip()]
    # The configured default is always offered, even if it was left out of the list.
    if settings.runpod_gpu_type_id and settings.runpod_gpu_type_id not in names:
        names.insert(0, settings.runpod_gpu_type_id)
    return names


def _headers() -> dict:
    if not settings.runpod_api_key:
        raise RunPodError(
            "RunPod is not configured on this server (RUNPOD_API_KEY is unset)."
        )
    return {"Authorization": f"Bearer {settings.runpod_api_key}"}


async def get_availability(gpu_type_id: str | None = None) -> dict:
    """Price and stock band for a GPU in the configured datacenter.

    Returns {"available": bool, "price": float|None, "stock": str|None, ...}.

    IMPORTANT: `available` means "RunPod quotes a price for this GPU here", which is NOT the same
    as "a pod can be placed". Measured 2026-08-08: community 4090 reported available=True,
    stock="Low" continuously while every single create failed with "this machine does not have
    the resources". lowestPrice knows nothing about whether any individual host has room. So
    treat this as a cheap negative signal — a False is a reliable no, a True is not a yes.

    Availability also genuinely flaps: sampling one GPU/DC pair ten times over a few minutes has
    returned stock in some samples and nothing in others.
    """
    gpu = gpu_type_id or settings.runpod_gpu_type_id
    # dataCenterId is omitted entirely when nothing is pinned. Passing null narrows the query
    # to "datacenters with no id", which reports no capacity anywhere — the failure would look
    # exactly like being out of stock.
    if settings.runpod_datacenter_id:
        query = """
        query ($gpu: String, $secure: Boolean, $dc: String) {
          gpuTypes(input: {id: $gpu}) {
            id
            lowestPrice(input: {gpuCount: 1, secureCloud: $secure, dataCenterId: $dc}) {
              uninterruptablePrice
              stockStatus
            }
          }
        }
        """
        variables = {
            "gpu": gpu,
            "secure": _is_secure(),
            "dc": settings.runpod_datacenter_id,
        }
    else:
        query = """
        query ($gpu: String, $secure: Boolean) {
          gpuTypes(input: {id: $gpu}) {
            id
            lowestPrice(input: {gpuCount: 1, secureCloud: $secure}) {
              uninterruptablePrice
              stockStatus
            }
          }
        }
        """
        variables = {"gpu": gpu, "secure": _is_secure()}

    payload = {"query": query, "variables": variables}
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.post(GRAPHQL, json=payload, headers=_headers())
    if resp.status_code == 401:
        raise RunPodError("RunPod rejected the API key (401).")
    if not resp.is_success:
        raise RunPodError(f"RunPod availability check failed ({resp.status_code}).")

    types = (resp.json().get("data") or {}).get("gpuTypes") or []
    lowest = (types[0].get("lowestPrice") or {}) if types else {}
    price = lowest.get("uninterruptablePrice")
    return {
        "gpu_type_id": gpu,
        "datacenter_id": settings.runpod_datacenter_id or "any",
        "cloud_type": settings.runpod_cloud_type,
        # No price means no inventory for this GPU in this datacenter right now.
        "available": price is not None,
        "price_per_hr": price,
        "stock": lowest.get("stockStatus"),
    }


async def launch_worker(name: str, env: dict[str, str], gpu_type_id: str | None = None) -> dict:
    """Create a pod, attached to the configured network volume.

    On community cloud there is no volume and no datacenter pin, so the pod lands wherever there
    is capacity and stages its own models. When a volume IS configured, the datacenter must be
    pinned to match it: a pod in the wrong region cannot mount it, and RunPod does not fail
    loudly about that — it just runs without the models and re-downloads ~39GB.

    gpu_type_id picks between the GPUs in selectable_gpus(); None uses the configured default.
    The caller is responsible for having validated it against that allowlist.
    """
    spec = {
        "name": name,
        "imageName": settings.runpod_image,
        "gpuTypeIds": [gpu_type_id or settings.runpod_gpu_type_id],
        "gpuCount": 1,
        "cloudType": settings.runpod_cloud_type,
        "containerDiskInGb": settings.runpod_container_disk_gb,
        "volumeMountPath": settings.runpod_volume_mount_path,
        "ports": ["8188/http", "22/tcp"],
        "env": env,
    }
    # Only sent when configured. Community cloud supports neither, and a network volume is
    # region-locked — so pinning a datacenter is meaningful only when there is a volume to sit
    # beside. Sending either on community would be rejected or silently ignored.
    # A network volume supplies the mount when configured; otherwise the pod needs its own disk.
    # Sending neither leaves volumeMountPath pointing at the container disk, which is not big
    # enough for the model set and fails during staging rather than at create time.
    if settings.runpod_network_volume_id:
        spec["networkVolumeId"] = settings.runpod_network_volume_id
    elif settings.runpod_volume_gb:
        spec["volumeInGb"] = settings.runpod_volume_gb
    if settings.runpod_datacenter_id:
        spec["dataCenterIds"] = [settings.runpod_datacenter_id]
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.post(f"{REST}/pods", json=spec, headers=_headers())

    if resp.status_code == 401:
        raise RunPodError("RunPod rejected the API key (401).")
    if not resp.is_success:
        # RunPod reports "no capacity" as a 4xx with prose. Pass it through rather than
        # inventing our own wording — theirs distinguishes no-stock from a bad spec.
        raise RunPodError(_readable_error(resp))

    body = resp.json()
    return {
        "id": body.get("id"),
        "name": body.get("name"),
        "status": body.get("desiredStatus"),
        "cost_per_hr": body.get("costPerHr"),
    }


async def list_workers() -> list[dict]:
    """Pods currently known to RunPod, launched by us or otherwise."""
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.get(f"{REST}/pods", headers=_headers())
    if not resp.is_success:
        raise RunPodError(f"Could not list RunPod pods ({resp.status_code}).")
    pods = resp.json()
    pods = pods if isinstance(pods, list) else pods.get("data", [])
    return [
        {
            "id": p.get("id"),
            "name": p.get("name"),
            "status": p.get("desiredStatus"),
            "cost_per_hr": p.get("costPerHr"),
            "gpu_type_id": (p.get("machine") or {}).get("gpuTypeId"),
        }
        for p in pods
    ]


async def terminate_worker(pod_id: str) -> None:
    """Terminate a pod outright.

    Terminate, not stop: a stopped pod stays EXITED and keeps billing for its disk. Callers are
    responsible for knowing the worker has finished its work — GPU or VRAM going idle is NOT
    that signal, because decode, RIFE, stitching, faceswap and identity scoring all run after
    the GPU goes quiet (47s to over 2 minutes, measured).
    """
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.delete(f"{REST}/pods/{pod_id}", headers=_headers())
    # 404 means it is already gone, which is the state the caller wanted.
    if resp.status_code == 404:
        return
    if not resp.is_success:
        raise RunPodError(_readable_error(resp))


def _readable_error(resp: httpx.Response) -> str:
    """Pull something a human can act on out of a RunPod error response."""
    try:
        body = resp.json()
    except Exception:
        return f"RunPod returned {resp.status_code}: {resp.text[:200]}"
    raw = ""
    if isinstance(body, dict):
        for key in ("error", "message", "detail"):
            value = body.get(key)
            if isinstance(value, str) and value:
                raw = value
                break
            if isinstance(value, dict) and value.get("message"):
                raw = str(value["message"])
                break
    if not raw:
        return f"RunPod returned {resp.status_code}: {str(body)[:200]}"
    return explain_placement_failure(raw)


# RunPod says two very different things in the same 500, and the difference is the whole
# diagnosis. Measured directly: on community, an A5000 request returns "no instances currently
# available" while a 4090 returns "does not have the resources" -- and the identical 4090 spec
# succeeds as interruptible. So the 4090 message means a host WAS matched and could not fit an
# on-demand reservation, not that the fleet is empty. Passing RunPod's wording through
# unexplained sent a user into dozens of blind retries against a request that could not succeed.
_NO_STOCK = "no instances currently available"
_NO_FIT = "does not have the resources"


def explain_placement_failure(message: str) -> str:
    """Turn RunPod's two placement errors into something that says what to do next."""
    low = message.lower()
    if _NO_FIT in low:
        return (
            f"{message.strip().rstrip('.')}. RunPod found a matching host but could not fit the "
            "pod on it — on community cloud the fleet is largely partially committed, so this is "
            "typical for in-demand GPUs and is not a spec problem. Retrying can work; picking a "
            "different GPU usually does."
        )
    if _NO_STOCK in low:
        return (
            f"{message.strip().rstrip('.')}. There is genuinely no stock for this GPU on this "
            "cloud right now — unlike a fit failure, retrying will not help until stock returns."
        )
    return message
