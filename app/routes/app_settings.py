from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import get_current_user
from app.database import get_db
from app.models import AppSetting, User
from app.schemas.app_settings import AppSettingsResponse, AppSettingsUpdate

router = APIRouter()

# Defaults if a key is missing from the DB
_DEFAULTS = {
    "cfg_high": "1",
    "cfg_low": "1",
    "lightx2v_strength_high": "2.0",
    "lightx2v_strength_low": "1.0",
    "negative_prompt": "",
}


async def _get_all_settings(db: AsyncSession) -> dict[str, str]:
    result = await db.execute(select(AppSetting))
    rows = {row.key: row.value for row in result.scalars().all()}
    return {k: rows.get(k, v) for k, v in _DEFAULTS.items()}


def _to_response(settings: dict[str, str]) -> AppSettingsResponse:
    return AppSettingsResponse(
        cfg_high=float(settings["cfg_high"]),
        cfg_low=float(settings["cfg_low"]),
        lightx2v_strength_high=float(settings["lightx2v_strength_high"]),
        lightx2v_strength_low=float(settings["lightx2v_strength_low"]),
        negative_prompt=settings["negative_prompt"],
    )


@router.get("/settings", response_model=AppSettingsResponse)
async def get_settings(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    settings = await _get_all_settings(db)
    return _to_response(settings)


@router.put("/settings", response_model=AppSettingsResponse)
async def update_settings(
    body: AppSettingsUpdate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    updates = body.model_dump(exclude_none=True)
    now = datetime.now(timezone.utc)
    for key, value in updates.items():
        existing = await db.get(AppSetting, key)
        if existing:
            existing.value = str(value)
            existing.updated_at = now
        else:
            db.add(AppSetting(key=key, value=str(value), updated_at=now))
    await db.commit()

    settings = await _get_all_settings(db)
    return _to_response(settings)
