from typing import Optional

from pydantic import BaseModel


class AppSettingsResponse(BaseModel):
    negative_prompt: str


class AppSettingsUpdate(BaseModel):
    negative_prompt: Optional[str] = None
