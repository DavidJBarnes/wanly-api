"""Fakes for the caption a ticket makes (wanly-console#572).

A ticket used to make a scene AND a motion paragraph; the motion half was removed in
wanly-api#481, so a ticket makes the scene alone, through captions.caption_scene. Tests still
say what the captioner "returns" with a ScenePair (now defined here: the app no longer has
one), so the shape of the old tests survives.
"""
from contextlib import contextmanager
from dataclasses import dataclass
from unittest.mock import AsyncMock, patch


@dataclass
class ScenePair:
    """What the captioner answers for one image: the scene and the instruction used."""
    scene: str
    scene_instruction: str = "i"


def scene_mock(pair=None, error=None):
    """A caption_scene mock answering as `pair` would have, or raising `error`."""
    if error is not None:
        return AsyncMock(side_effect=error)
    return AsyncMock(return_value=(pair.scene, pair.scene_instruction))


@contextmanager
def ticket_captioner(pair=None, error=None):
    scene = scene_mock(pair, error)
    with patch("app.routes.captions.caption_scene", scene):
        yield scene


def drive_scene_from(monkeypatch, pair_mock):
    """Keep a test's `.pair` mock meaningful: it is awaited once per scene a ticket makes, so
    its call counts, return values and side effects mean what they always did."""

    async def scene(image, cfg, fallback_base, style=None, instruction=None):
        from app.joycaption import scene_captioner
        if scene_captioner() is None:
            # Unsplit (the default in tests): the scene goes to the shared captioner, through
            # its busy-checked base, exactly as the real caption_scene does.
            await fallback_base()
        pair = await pair_mock(image, "http://c", cfg, style=style, instruction=instruction)
        return pair.scene, pair.scene_instruction

    monkeypatch.setattr("app.routes.captions.caption_scene", scene)
