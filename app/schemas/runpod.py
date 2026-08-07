from pydantic import BaseModel


class RunPodAvailability(BaseModel):
    gpu_type_id: str
    datacenter_id: str
    available: bool
    price_per_hr: float | None = None
    stock: str | None = None


class RunPodWorker(BaseModel):
    id: str
    name: str | None = None
    status: str | None = None
    cost_per_hr: float | None = None
    gpu_type_id: str | None = None


class RunPodLaunchRequest(BaseModel):
    name: str
    # Override only when pointing a worker at something other than the configured queue —
    # e.g. testing against a staging API. Normal launches leave it unset.
    queue_url: str | None = None
