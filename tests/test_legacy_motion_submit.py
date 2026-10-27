"""A legacy <MOTION> at submit (wanly-api#481).

The motion caption half is gone. An old pose that still carries <MOTION> must never be held
for it and never reach the encoder as literal text: it is stripped before the empty-prompt
checks, so a recipe that was only "<trigger>, <MOTION>" is refused with a reason that says
why, and one with words of its own goes through without the placeholder.
"""
import pytest
from fastapi import HTTPException

from app.caption_hold import fill_saved, needed_halves
from app.models import ImageMeta
from app.routes.segments import _refuse_empty_submit, _strip_legacy_motion

RECIPE = {"characters": [{"trigger": "k3lly2026", "gender": "woman", "char_lora": "k.safetensors"}]}


@pytest.mark.asyncio
class TestRefuseEmptySubmit:
    async def test_trigger_plus_motion_is_refused_and_says_motion_is_gone(self):
        with pytest.raises(HTTPException) as e:
            await _refuse_empty_submit(None, "k3lly2026, woman, <MOTION>", RECIPE)
        assert e.value.status_code == 422
        assert "only the character's trigger phrase" in e.value.detail
        assert "<MOTION> is no longer supported" in e.value.detail

    async def test_motion_alone_is_refused_as_empty(self):
        with pytest.raises(HTTPException) as e:
            await _refuse_empty_submit(None, "<MOTION>", RECIPE)
        assert "empty" in e.value.detail and "<MOTION>" in e.value.detail

    async def test_a_filled_motion_region_alone_is_refused_too(self):
        """Its words came from the retired half: stripped with it, so they cannot carry a
        prompt on their own."""
        with pytest.raises(HTTPException):
            await _refuse_empty_submit(
                None, "k3lly2026, woman, <motion>she rocks forward</motion>", RECIPE)

    async def test_words_of_its_own_pass(self):
        await _refuse_empty_submit(None, "k3lly2026, woman, she laughs, <MOTION>", RECIPE)

    async def test_scene_plus_motion_passes_on_the_scene(self):
        await _refuse_empty_submit(None, "k3lly2026, woman, <SCENE>, <MOTION>", RECIPE)


class TestNeverHeld:
    def test_motion_is_not_a_half_to_wait_for(self):
        assert needed_halves("k3lly2026, woman, <MOTION>") == set()
        assert needed_halves("k3lly2026, woman, <SCENE>, <MOTION>") == {"scene"}

    def test_release_fills_the_scene_and_strips_the_motion(self):
        meta = ImageMeta(path="s3://b/x.png", scene_description="a woman on a sofa")
        out = fill_saved("k3lly2026, woman, <SCENE>, <MOTION>", meta)
        assert out == "k3lly2026, woman, a woman on a sofa"

    def test_release_with_no_scene_keeps_the_placeholder_but_not_the_motion(self):
        out = fill_saved("k3lly2026, <SCENE>, <MOTION>", None)
        assert out == "k3lly2026, <SCENE>"


@pytest.mark.parametrize("prompt, expected", [
    ("<MOTION>", ""),
    ("k3lly2026, woman, <MOTION>", "k3lly2026, woman"),
    ("<MOTION>, she laughs", "she laughs"),
    ("a, <MOTION>, b", "a, b"),
    ("She smiles. <MOTION>", "She smiles."),
    ("<MOTION>. She smiles", "She smiles"),
    ("a, <motion>she rocks</motion>, b", "a, b"),
    ("a\n\n<MOTION>\nb", "a b"),
])
def test_strip_is_tidy(prompt, expected):
    assert _strip_legacy_motion(prompt) == expected


def test_a_prompt_without_motion_is_returned_untouched():
    p = "k3lly2026,  woman ,\n<SCENE>"
    assert _strip_legacy_motion(p) is p
