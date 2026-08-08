from pydantic import BaseModel


class SegmentRuntimeGroup(BaseModel):
    """One GPU + job shape combination.

    Shape is part of the identity, not a detail: the same hardware runs 480p/3s in ~330s and
    720x1056/5s in ~1780s, so a figure that mixes them describes nothing.
    """

    gpu_name: str
    width: int
    height: int
    clip_seconds: float
    samples: int
    avg_seconds: float
    median_seconds: float
    min_seconds: float
    max_seconds: float
