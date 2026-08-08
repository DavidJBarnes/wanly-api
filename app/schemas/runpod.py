from pydantic import BaseModel


class RunPodAvailability(BaseModel):
    gpu_type_id: str
    datacenter_id: str
    available: bool
    price_per_hr: float | None = None
    stock: str | None = None
    cloud_type: str | None = None


class RunPodWorker(BaseModel):
    id: str
    name: str | None = None
    status: str | None = None
    cost_per_hr: float | None = None
    gpu_type_id: str | None = None


class RunPodGpuOption(BaseModel):
    """One selectable GPU, with its live price and stock band."""

    gpu_type_id: str
    is_default: bool
    available: bool
    price_per_hr: float | None = None
    stock: str | None = None
    # Set when the price/stock lookup itself failed, so the UI can show the option as
    # selectable-but-unknown rather than silently reporting it as unavailable.
    error: str | None = None


class RunPodLaunchRequest(BaseModel):
    name: str
    # Override only when pointing a worker at something other than the configured queue —
    # e.g. testing against a staging API. Normal launches leave it unset.
    queue_url: str | None = None
    # Which GPU to ask RunPod for. None uses the configured default. Validated against the
    # server's allowlist — an unsupported GPU (the 5090 does not run our workflow) must not be
    # reachable by sending an arbitrary string.
    gpu_type_id: str | None = None
