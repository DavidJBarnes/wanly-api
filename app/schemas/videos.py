from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel


class VideoResponse(BaseModel):
    id: UUID
    job_id: UUID
    output_path: Optional[str]
    duration_seconds: Optional[float]
    status: str
    error_message: Optional[str]
    created_at: datetime
    completed_at: Optional[datetime]

    model_config = {"from_attributes": True}
