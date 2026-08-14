from typing import Optional

from pydantic import BaseModel


class AppSettingsResponse(BaseModel):
    cfg_high: float
    cfg_low: float
    lightx2v_strength_high: float
    lightx2v_strength_low: float
    steps_total: int
    high_noise_steps: int
    flow_shift: float
    negative_prompt: str
    # Segment continuation strategy: "traditional" (last-frame i2v handoff, current)
    # or "vace" (video-conditioned continuation). vace_overlap_frames = kept tail length.
    continuation_mode: str = "traditional"
    vace_overlap_frames: int = 12
    # "Re-roll until": how many takes a rule-driven re-roll chain may generate per job
    # before it gives up and sits idle (the user-initiated roll counts as the first).
    max_rerolls_per_job: int = 3
    # Seed re-anchor: faceswap each continuation segment's last frame to the canonical
    # identity before it seeds the next segment (only fires when a successor exists).


class AppSettingsUpdate(BaseModel):
    cfg_high: Optional[float] = None
    cfg_low: Optional[float] = None
    lightx2v_strength_high: Optional[float] = None
    lightx2v_strength_low: Optional[float] = None
    steps_total: Optional[int] = None
    high_noise_steps: Optional[int] = None
    flow_shift: Optional[float] = None
    negative_prompt: Optional[str] = None
    continuation_mode: Optional[str] = None
    vace_overlap_frames: Optional[int] = None
    max_rerolls_per_job: Optional[int] = None
