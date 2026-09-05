"""The default negative prompt, resolved in ONE place.

There used to be two answers to "what negative prompt does a render use when nothing sets
one", and they disagreed. `LTX_STACK['negative']` is a code constant baked into the image;
`app_settings.negative_prompt` is the Settings page field. The recipe book resolved a pose
against the constant, the console prefilled its form from that, and every segment was
therefore created carrying it -- so the claim endpoint's fallback to the setting, the only
place the setting was ever read, could not fire. The field was dead by construction
(console#430).

Resolution order is pose override, then the setting, then the constant. The constant is
demoted to a SEED: it is what a fresh install renders with before anyone opens Settings,
not a default that outranks a stated preference.
"""

from sqlalchemy.ext.asyncio import AsyncSession

from app.ltx_stack import LTX_STACK
from app.models import AppSetting

SETTING_KEY = "negative_prompt"


async def default_negative_prompt(db: AsyncSession) -> str:
    """What a segment or pose that sets no negative prompt of its own renders with.

    Blank means "not set", not "render with no negative at all" -- an empty text box is how
    the field looks before it is ever used, and reading that as "drop the quality negatives"
    would silently change every render the first time someone cleared it. Whitespace counts
    as blank for the same reason.
    """
    row = await db.get(AppSetting, SETTING_KEY)
    configured = (row.value if row else None) or ""
    return configured if configured.strip() else LTX_STACK["negative"]
