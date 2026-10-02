"""Fakes for the two halves a caption ticket makes (wanly-console#572).

A ticket used to make its pair through one call, run_caption_pair, and tests patched that with
a ScenePair. Since the split the scene is made in the scene lane (captions.caption_scene) and
the motion paragraph in the motion lane (caption_tickets.describe_motion); this patches both
from the same ScenePair, so a test still says what the captioner "returns" in one place.
"""
from contextlib import contextmanager
from unittest.mock import AsyncMock, patch

from app.joycaption import CaptionError


def halves(pair=None, error=None):
    """(scene_mock, motion_mock) answering as `pair` would have, or raising `error`."""
    if error is not None:
        scene = AsyncMock(side_effect=error)
    else:
        scene = AsyncMock(return_value=(pair.scene, pair.scene_instruction))

    async def _motion(image, scene_text, style="", custom="", base_url=None):
        if pair is not None and pair.motion:
            return pair.motion, pair.motion_instruction
        raise CaptionError((pair.motion_error if pair is not None else None)
                           or "the captioner returned no motion paragraph")
    return scene, AsyncMock(side_effect=_motion)


@contextmanager
def ticket_captioner(pair=None, error=None):
    scene, motion = halves(pair, error)
    with patch("app.routes.captions.caption_scene", scene), \
         patch("app.caption_tickets.describe_motion", motion):
        yield scene


def drive_halves_from(monkeypatch, pair_mock, motion_mock):
    """Keep a test's `.pair` mock meaningful after the split.

    `pair_mock` is awaited once per scene the ticket makes (so its call counts, return values
    and side effects mean what they always did), and the motion half of the pair it returned
    answers that same ticket's motion step. A motion step with no pair behind it -- a
    motion-only ticket grounded on a saved scene -- goes to `motion_mock`, as before.
    """
    pending: dict[str, object] = {}

    async def scene(image, cfg, fallback_base, style=None, instruction=None):
        from app.joycaption import scene_captioner
        if scene_captioner() is None:
            # Unsplit (the default in tests): the scene goes to the shared captioner, through
            # its busy-checked base, exactly as the real caption_scene does.
            await fallback_base()
        pair = await pair_mock(image, "http://c", cfg, style=style, instruction=instruction)
        pending[pair.scene.strip()] = pair
        return pair.scene, pair.scene_instruction

    async def motion(image, scene_text, style="", custom="", base_url=None):
        pair = pending.pop((scene_text or "").strip(), None)
        if pair is None:
            return await motion_mock(image, scene_text, style=style, custom=custom,
                                     base_url=base_url)
        if getattr(pair, "motion_busy", False):
            from app.joycaption import CaptionerBusy
            raise CaptionerBusy(pair.motion_error or "busy")
        if pair.motion:
            return pair.motion, pair.motion_instruction
        raise CaptionError(pair.motion_error or "the captioner returned no motion paragraph")

    monkeypatch.setattr("app.routes.captions.caption_scene", scene)
    monkeypatch.setattr("app.caption_tickets.describe_motion", motion)
