"""POST /lora-usage: LoRA use reported from outside Wanly (wanly-api#458).

The reporter on the A1111 box (wanly-gpu-docker#211) sends per-name TOTALS on every run, so
this is an idempotent upsert -- the row becomes what was sent, it never adds. Worker auth
(X-API-Key), like every other box-to-API call.
"""
from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import verify_api_key
from app.database import get_db
from app.models import LoraUsage

router = APIRouter(tags=["lora-usage"])


class LoraUsageItem(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    source: str = Field(default="a1111", min_length=1, max_length=32)
    images: int = Field(ge=0)
    first_used_at: datetime | None = None
    last_used_at: datetime | None = None


class LoraUsageReport(BaseModel):
    items: list[LoraUsageItem] = Field(max_length=20_000)


@router.post("/lora-usage", dependencies=[Depends(verify_api_key)])
async def report_lora_usage(body: LoraUsageReport, db: AsyncSession = Depends(get_db)):
    now = datetime.now(timezone.utc)
    # One name may appear once per source; the last wins if a reporter repeats it.
    rows = {(i.name.strip(), i.source): i for i in body.items if i.name.strip()}
    if rows:
        stmt = insert(LoraUsage).values([
            {"name": n, "source": s, "images": i.images, "first_used_at": i.first_used_at,
             "last_used_at": i.last_used_at, "updated_at": now}
            for (n, s), i in rows.items()])
        stmt = stmt.on_conflict_do_update(
            constraint="uq_lora_usage_name_source",
            set_={"images": stmt.excluded.images, "first_used_at": stmt.excluded.first_used_at,
                  "last_used_at": stmt.excluded.last_used_at, "updated_at": now})
        await db.execute(stmt)
        await db.commit()
    return {"stored": len(rows)}
