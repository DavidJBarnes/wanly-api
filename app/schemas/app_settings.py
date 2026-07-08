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


class AppSettingsUpdate(BaseModel):
    cfg_high: Optional[float] = None
    cfg_low: Optional[float] = None
    lightx2v_strength_high: Optional[float] = None
    lightx2v_strength_low: Optional[float] = None
    steps_total: Optional[int] = None
    high_noise_steps: Optional[int] = None
    flow_shift: Optional[float] = None
    negative_prompt: Optional[str] = None
