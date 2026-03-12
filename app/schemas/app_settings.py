from typing import Optional

from pydantic import BaseModel


class AppSettingsResponse(BaseModel):
    cfg_high: float
    cfg_low: float
    lightx2v_strength_high: float
    lightx2v_strength_low: float


class AppSettingsUpdate(BaseModel):
    cfg_high: Optional[float] = None
    cfg_low: Optional[float] = None
    lightx2v_strength_high: Optional[float] = None
    lightx2v_strength_low: Optional[float] = None
