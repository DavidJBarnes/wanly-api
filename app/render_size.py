"""The size a recipe clip actually renders at, which is not always the job's size (#359).

A job's width/height are its START FRAME's size, and they stay that way: they describe the
image the user picked. Since wanly-gpu-docker#148 the engine does not render at that size for a
recipe, though. It derives the clip size from the start frame and caps it at an area, so an
upscaled 1856x1280 frame renders at 1216x832. Anything that prices or describes the render -
the run-time estimate, the queue ETA, the job page - has to use the size that renders, or it
reports a clip roughly twice as big as the one the GPU is working on.

THE ENGINE IS THE SOURCE OF TRUTH. `effective_render_size` is a copy of derive_size() in
wanly-gpu-docker's engine/app.py, and tests/test_render_size.py pins it to the same cases as
that repo's tests/test_render_cap.py. Change one and the other has to move with it.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import func

from app.config import settings
from app.models import Segment


def effective_render_size(
    width: int, height: int, max_pixels: int | None = None
) -> tuple[int, int]:
    """The engine's derive_size(), on a size rather than an image file.

    Scale DOWN (never up) to fit `max_pixels` keeping the aspect, then round down to /64, with
    64 as the floor. `max_pixels` defaults to settings.max_render_pixels; 0 means no cap, which
    still snaps to the /64 grid because the engine does.
    """
    cap = settings.max_render_pixels if max_pixels is None else max_pixels
    w, h = width, height
    if cap and w * h > cap:
        scale = (cap / (w * h)) ** 0.5
        # round() before flooring, exactly as the engine does: 1824 * 0.6667 is 1215.99..., and
        # flooring that straight to /64 would drop a whole step to 1152.
        w, h = round(w * scale), round(h * scale)
    return max(64, (w // 64) * 64), max(64, (h // 64) * 64)


def renders_through_recipe(ltx_recipe: dict[str, Any] | None) -> bool:
    """Whether a segment takes the engine's recipe path, the only one that derives its size.

    The daemon forwards `recipe` to the engine only when it is set (ltx_client's
    build_submit_payload), and the engine derives the size only when it is. A NULL blob is a
    WAN or free-form LTX segment, which renders at exactly the size it was asked for.

    A recipe with no start frame (the regularization pool) renders text-to-video at the
    request's size instead. Not modelled: those jobs are 1216x832 or 832x1216, which the rule
    leaves unchanged, so treating them as derived gives the same answer.
    """
    return bool(ltx_recipe and ltx_recipe.get("recipe"))


#: renders_through_recipe() as SQL, for queries that never load the blob. `->>` gives NULL for a
#: missing key and for JSON null alike, and an empty name is not a recipe either.
RECIPE_SQL = func.coalesce(Segment.ltx_recipe["recipe"].astext, "") != ""


def render_size(width: int, height: int, recipe: bool) -> tuple[int, int]:
    """The size a segment renders at: derived for a recipe, as stored for anything else."""
    return effective_render_size(width, height) if recipe else (width, height)
