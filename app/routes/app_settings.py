import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import get_current_user
from app.database import get_db
from app.joycaption import CAPTION_STYLES, DEFAULT_STYLE
from app.models import AppSetting, User
from app.schemas.app_settings import AppSettingsResponse, AppSettingsUpdate

logger = logging.getLogger(__name__)

router = APIRouter()

# Defaults if a key is missing from the DB
_DEFAULTS = {
    "negative_prompt": "",
    # See app/joycaption.py for what each style asks for. "standard" is the one that tested
    # best on real frames: ~40 words, and it explicitly requests gaze and expression, which
    # a plain "describe this image" omits and which carry real weight in an LTX prompt.
    "caption_style": DEFAULT_STYLE,
    # Empty means "use the style". A non-empty value wins over it.
    "caption_instruction": "",
}


async def _get_all_settings(db: AsyncSession) -> dict[str, str]:
    result = await db.execute(select(AppSetting))
    rows = {row.key: row.value for row in result.scalars().all()}
    return {k: rows.get(k, v) for k, v in _DEFAULTS.items()}


def _to_response(settings: dict[str, str]) -> AppSettingsResponse:
    style = settings.get("caption_style") or DEFAULT_STYLE
    if style not in CAPTION_STYLES:
        # A style written directly into the table, or one removed in a later release. Fall
        # back rather than 500 the whole settings page over one bad row.
        logger.warning("unknown caption_style %r in app_settings; using %r", style, DEFAULT_STYLE)
        style = DEFAULT_STYLE
    return AppSettingsResponse(
        negative_prompt=settings["negative_prompt"],
        caption_style=style,
        caption_instruction=settings.get("caption_instruction", ""),
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
