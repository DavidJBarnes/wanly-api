import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import get_current_user
from app.database import get_db
from app.joycaption import CAPTION_STYLES, DEFAULT_STYLE
from app.models import AppSetting, User
from app import queue_position
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
    # (motion_style / motion_instruction were the retired motion caption half's settings,
    # wanly-api#481. Rows left in app_settings by older releases are simply never read.)
    # Where a newly queued segment lands (#471): "top" keeps its job's slot, "bottom" sends
    # the job behind everything else. See app/queue_position.py.
    queue_position.SETTING_KEY: queue_position.DEFAULT,
}


def _unpin_defaults(updates: dict, current: dict[str, str]) -> dict:
    """Store "" for an override that is word-for-word the default it would replace.

    The Settings editors are pre-filled with the default text (console#555), so saving the
    page untouched sends that text back. Stored as-is it would be an override that happens
    to match today -- and would quietly pin the old wording when a default is next improved,
    which is the opposite of "defaults unchanged for anyone who doesn't touch them". The
    console already sends "" in that case; this is the guard for any caller that does not.

    The caption comparison is against the style that will be in force after this update,
    since that is the preset an empty instruction falls back to.
    """
    out = dict(updates)
    style = out.get("caption_style") or current.get("caption_style") or DEFAULT_STYLE
    caption = out.get("caption_instruction")
    if caption and caption.strip() == CAPTION_STYLES.get(style, "").strip():
        out["caption_instruction"] = ""
    return out


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
        segment_queue_position=(
            queue_position.BOTTOM
            if settings.get(queue_position.SETTING_KEY) == queue_position.BOTTOM
            else queue_position.TOP),
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
    updates = _unpin_defaults(body.model_dump(exclude_none=True), await _get_all_settings(db))
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
