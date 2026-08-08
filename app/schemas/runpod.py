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
    # What we asked RunPod for. Not the same as what is running -- see runtime_ready.
    status: str | None = None
    cost_per_hr: float | None = None
    gpu_type_id: str | None = None
    created_at: str | None = None
    # False means the pod is rented and billing but its container is not up. A pod can sit like
    # this indefinitely; it is the difference between "still booting" and "never going to boot".
    runtime_ready: bool = False
    # Zero on a running pod means the host never attached a GPU, which is fatal and immediate --
    # torch sees no devices and ComfyUI dies on import.
    gpu_count: int = 0


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
