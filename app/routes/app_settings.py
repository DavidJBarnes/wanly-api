import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import get_current_user
from app.database import get_db
from app.joycaption import (CAPTION_STYLES, DEFAULT_STYLE, MOTION_DEFAULT_STYLE,
                            MOTION_INSTRUCTIONS, MOTION_STYLE_PRESETS, resolve_motion_override,
                            strip_motion_tags)
from app.models import AppSetting, User
from app.schemas.app_settings import AppSettingsResponse, AppSettingsUpdate

logger = logging.getLogger(__name__)

router = APIRouter()

#: Where the motion editor's plain instructions are stored (console#573).
#:
#: A NEW KEY, NOT A NEW MEANING FOR THE OLD ONE. "motion_instruction" already holds one of two
#: things: a #555 template (tagged; migrates to its instruction text) or a pre-#555 whole
#: prompt (untagged; must keep meaning "send exactly this"). Plain instructions are untagged
#: too, so in the old key they would be indistinguishable from a whole prompt -- detecting
#: tags on read can tell a template from prose, but not prose-to-wrap from prose-to-send.
#: The key a value lives in says which it is, with no flag to keep in step. Saving from the
#: editor writes here and clears the old key, so the legacy reading ends the first time the
#: person saves the motion prompt.
MOTION_KEY = "motion_prompt_instructions"
#: The pre-#573 key, read for migration and cleared on save. Never written with text again.
LEGACY_MOTION_KEY = "motion_instruction"

# Defaults if a key is missing from the DB
_DEFAULTS = {
    "negative_prompt": "",
    # See app/joycaption.py for what each style asks for. "standard" is the one that tested
    # best on real frames: ~40 words, and it explicitly requests gaze and expression, which
    # a plain "describe this image" omits and which carry real weight in an LTX prompt.
    "caption_style": DEFAULT_STYLE,
    # Empty means "use the style". A non-empty value wins over it.
    "caption_instruction": "",
    # The motion half (#326): how the capture should look when the frame is described as a
    # 10-second clip. "handheld" is the house style measured in the prototype.
    "motion_style": MOTION_DEFAULT_STYLE,
    # Empty means "use the default instructions" (joycaption.MOTION_INSTRUCTIONS). The API
    # adds the style sentence and the grounding around them (console#573).
    MOTION_KEY: "",
    LEGACY_MOTION_KEY: "",
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
    motion = out.get("motion_instruction")
    if motion and strip_motion_tags(motion).strip() == MOTION_INSTRUCTIONS.strip():
        out["motion_instruction"] = ""
    return out


async def _get_all_settings(db: AsyncSession) -> dict[str, str]:
    """Every setting, with the motion override already resolved (console#573).

    "motion_instruction" is the plain instructions in force ("" = the default) and
    "motion_legacy_prompt" a pre-#555 whole prompt still in force ("" = none), whichever
    key they were read from. Every consumer reads these two, so none of them needs to know
    there are two keys.
    """
    result = await db.execute(select(AppSetting))
    rows = {row.key: row.value for row in result.scalars().all()}
    cfg = {k: rows.get(k, v) for k, v in _DEFAULTS.items()}
    instructions, legacy = resolve_motion_override(cfg.pop(MOTION_KEY),
                                                   cfg.pop(LEGACY_MOTION_KEY))
    cfg["motion_instruction"] = instructions
    cfg["motion_legacy_prompt"] = legacy
    return cfg


def _to_response(settings: dict[str, str]) -> AppSettingsResponse:
    style = settings.get("caption_style") or DEFAULT_STYLE
    if style not in CAPTION_STYLES:
        # A style written directly into the table, or one removed in a later release. Fall
        # back rather than 500 the whole settings page over one bad row.
        logger.warning("unknown caption_style %r in app_settings; using %r", style, DEFAULT_STYLE)
        style = DEFAULT_STYLE
    motion_style = settings.get("motion_style") or MOTION_DEFAULT_STYLE
    if motion_style not in MOTION_STYLE_PRESETS:
        logger.warning("unknown motion_style %r in app_settings; using %r",
                       motion_style, MOTION_DEFAULT_STYLE)
        motion_style = MOTION_DEFAULT_STYLE
    return AppSettingsResponse(
        negative_prompt=settings["negative_prompt"],
        caption_style=style,
        caption_instruction=settings.get("caption_instruction", ""),
        motion_style=motion_style,
        motion_instruction=settings.get("motion_instruction", ""),
        motion_legacy_prompt=settings.get("motion_legacy_prompt", ""),
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
    if "motion_instruction" in updates:
        # The editor's instructions go to the new key, and the old one is cleared: whatever
        # it held -- a #555 template or a whole prompt -- is replaced by what was just saved.
        # Omitting motion_instruction leaves both alone, which is how the console keeps a
        # legacy whole prompt it has not been asked to change.
        updates[MOTION_KEY] = updates.pop("motion_instruction")
        updates[LEGACY_MOTION_KEY] = ""
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
