"""LTX stage 2's noise schedule, a global from Settings (console#717).

The engine's graph template carries the validated schedule, which `LTX_STACK` records as the
default. Blank means "use it": the claim then sends nothing, so the graph and its hash are
exactly what they were before this setting existed.
"""

import re

from sqlalchemy.ext.asyncio import AsyncSession

from app.ltx_stack import LTX_STACK
from app.models import AppSetting

SETTING_KEY = "stage2_sigmas"
DEFAULT = LTX_STACK["sigmas_stage_2"]

MAX_VALUES = 20
# Plain decimals only: ComfyUI's ManualSigmas pulls numbers out with a regex, so `1e-3` or a
# stray word would be misread rather than refused.
_DECIMAL = re.compile(r"\d*\.?\d+")


def parse(text: str) -> list[float]:
    """The schedule as numbers, or ValueError naming what is wrong with it."""
    parts = [p.strip() for p in text.split(",")]
    if not all(_DECIMAL.fullmatch(p) for p in parts):
        raise ValueError("sigmas must be plain decimals separated by commas")
    values = [float(p) for p in parts]
    if not 2 <= len(values) <= MAX_VALUES:
        raise ValueError(f"sigmas must have between 2 and {MAX_VALUES} values")
    if values[0] > 1:
        raise ValueError("sigmas must start at 1 or below")
    if any(a <= b for a, b in zip(values, values[1:])):
        raise ValueError("sigmas must strictly decrease")
    if values[-1] != 0:
        raise ValueError("sigmas must end at 0")
    return values


def normalise(text: str) -> str:
    """Validated and tidied for storage. Blank stays blank: that is how it is cleared."""
    if not text.strip():
        return ""
    parse(text)
    return ", ".join(p.strip() for p in text.split(","))


def is_default(text: str) -> bool:
    return bool(text.strip()) and parse(text) == parse(DEFAULT)


async def configured_stage2_sigmas(db: AsyncSession) -> str | None:
    """The schedule Settings asks for, or None to leave the engine's own."""
    row = await db.get(AppSetting, SETTING_KEY)
    return ((row.value if row else "") or "").strip() or None
