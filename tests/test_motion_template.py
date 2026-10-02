"""The caption and motion prompts are editable from Settings (console#555).

Three promises carry the change, and each is pinned here:

1. NOBODY WHO DOES NOT TOUCH THE EDITORS SEES A DIFFERENT PROMPT. The motion prompt became a
   template; rendering the default template must give byte-for-byte what the hand-assembled
   prompt gave, with a scene and without one, for every capture style. The findings in
   app/joycaption.py above MOTION_BASE are why the exact words matter.
2. AN OVERRIDE WRITTEN BEFORE #555 KEEPS ITS MEANING: the whole prompt, no grounding.
3. A BAD TEMPLATE IS REFUSED ON THE WAY IN (422), and "try" stores nothing.
"""
from unittest.mock import patch

import pytest

from app.joycaption import (CAPTION_STYLES, MOTION_BASE, MOTION_DEFAULT_STYLE,
                            MOTION_GROUNDING, MOTION_STYLE_PRESETS, MOTION_TAIL,
                            MOTION_TEMPLATE, PROMPT_MAX_CHARS, motion_instruction_for,
                            render_motion_template, validate_motion_template)
from app.models import AppSetting, ImageMeta

BUCKET = "wanly-images"
PATH = f"s3://{BUCKET}/2026-09-29/00001.png"
SCENE = "A woman in a red dress sits on a sofa, holding a glass, looking at the viewer"


def _pre_555(style: str, custom: str = "", scene: str = "") -> str:
    """motion_instruction_for as it was before the template, copied verbatim.

    Kept as a literal copy rather than imported from history so the comparison is against
    what production actually sent, not against whatever the constants say now.
    """
    if custom and custom.strip():
        return custom.strip()
    style_sentence = MOTION_STYLE_PRESETS.get(style, MOTION_STYLE_PRESETS[MOTION_DEFAULT_STYLE])
    prompt = MOTION_BASE + style_sentence + MOTION_TAIL
    if scene and scene.strip():
        prompt = (f"Scene: {scene.strip()}\n\n{prompt}\n\n{MOTION_GROUNDING} "
                  "Do not restate the scene description.")
    return prompt


ALL_STYLES = [*MOTION_STYLE_PRESETS, "not-a-style"]


class TestTheDefaultIsUnchanged:
    @pytest.mark.parametrize("style", ALL_STYLES)
    def test_without_a_scene(self, style):
        assert motion_instruction_for(style) == _pre_555(style)

    @pytest.mark.parametrize("style", ALL_STYLES)
    def test_with_a_scene(self, style):
        assert motion_instruction_for(style, "", SCENE) == _pre_555(style, "", SCENE)

    @pytest.mark.parametrize("scene", ["", "   ", "\n\t"])
    def test_a_blank_scene_is_no_scene(self, scene):
        """Today a whitespace scene dropped the grounding; so must the template."""
        assert motion_instruction_for("handheld", "", scene) == _pre_555("handheld", "", scene)
        assert "Scene:" not in motion_instruction_for("handheld", "", scene)

    def test_a_scene_with_padding_is_stripped(self):
        assert (motion_instruction_for("amateur", "", f"  {SCENE}\n")
                == _pre_555("amateur", "", f"  {SCENE}\n"))

    def test_the_default_template_rendered_directly(self):
        """The template itself, not just motion_instruction_for, is the thing the console
        pre-fills and a user saves -- so saving it untouched must also be a no-op."""
        style = MOTION_STYLE_PRESETS["handheld"]
        assert render_motion_template(MOTION_TEMPLATE, style, SCENE) == _pre_555("handheld", "", SCENE)
        assert render_motion_template(MOTION_TEMPLATE, style) == _pre_555("handheld")
        # And a saved copy of the default goes down the custom path with the same result.
        assert motion_instruction_for("cinematic", MOTION_TEMPLATE, SCENE) == \
            _pre_555("cinematic", "", SCENE)

    def test_the_default_template_is_valid(self):
        assert validate_motion_template(MOTION_TEMPLATE) == MOTION_TEMPLATE

    def test_it_fits_under_the_cap(self):
        assert len(MOTION_TEMPLATE) < PROMPT_MAX_CHARS
        assert all(len(v) < PROMPT_MAX_CHARS for v in CAPTION_STYLES.values())


class TestLegacyOverrides:
    """A custom motion instruction saved before #555 has no placeholders."""

    def test_it_is_still_the_whole_prompt(self):
        legacy = "Describe the motion in this frame as a ten second clip."
        assert motion_instruction_for("handheld", legacy, SCENE) == legacy
        assert motion_instruction_for("handheld", legacy, SCENE) == _pre_555("handheld", legacy, SCENE)

    def test_no_grounding_is_added(self):
        out = motion_instruction_for("handheld", "  my own words  ", SCENE)
        assert out == "my own words"
        assert SCENE not in out and MOTION_GROUNDING not in out

    def test_literal_braces_that_are_not_placeholders_survive(self):
        text = "Use JSON-ish {a b} notes and a lone { brace."
        assert motion_instruction_for("handheld", text, SCENE) == text
        assert validate_motion_template(text) == text


class TestTemplateRendering:
    def test_an_edit_keeps_grounding_while_it_keeps_the_section(self):
        t = "{#scene}Scene: {scene}\n\n{/scene}Move. {style}"
        assert render_motion_template(t, "Handheld. ", SCENE) == f"Scene: {SCENE}\n\nMove. Handheld."
        assert render_motion_template(t, "Handheld. ") == "Move. Handheld."

    def test_scene_outside_a_section_is_empty_without_a_scene(self):
        assert render_motion_template("A {scene} B", "") == "A  B"

    def test_a_scene_containing_placeholder_text_is_not_expanded(self):
        """One substitution pass: the caption is inserted as text."""
        out = render_motion_template("{scene} / {style}", "S", "she says {style}")
        assert out == "she says {style} / S"

    def test_the_none_style_is_empty(self):
        assert render_motion_template("a{style}b", MOTION_STYLE_PRESETS["none"]) == "ab"

    def test_rendering_tolerates_what_validation_would_refuse(self):
        """A stored value must never 500 a caption; unknowns are left as written."""
        assert render_motion_template("x {sceen} {#scene}y", "") == "x {sceen} {#scene}y"


class TestValidation:
    @pytest.mark.parametrize("bad, fragment", [
        ("Describe {sceen}.", "{sceen}"),
        ("{Scene} and {style}", "{Scene}"),
        ("{#scene}unclosed", "never closed"),
        ("stray {/scene}", "without a matching"),
        ("{#scene}a{#scene}b{/scene}{/scene}", "nested"),
        ("{#style}x{/style}", "{#style}"),
    ])
    def test_refused(self, bad, fragment):
        with pytest.raises(ValueError) as e:
            validate_motion_template(bad)
        assert fragment in str(e.value)

    def test_two_sections_are_fine(self):
        t = "{#scene}a {scene}{/scene} middle {#scene}b{/scene}"
        assert validate_motion_template(t) == t


# ---------------------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------------------


async def _client(db):
    from httpx import ASGITransport, AsyncClient
    from app.auth import get_current_user
    from app.database import get_db
    from app.main import app

    app.dependency_overrides[get_current_user] = lambda: object()
    app.dependency_overrides[get_db] = lambda: db
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test"), app


async def _put(db, body):
    client, app = await _client(db)
    try:
        async with client as c:
            return await c.put("/settings", json=body)
    finally:
        app.dependency_overrides.clear()


async def _get(db):
    client, app = await _client(db)
    try:
        async with client as c:
            return await c.get("/settings")
    finally:
        app.dependency_overrides.clear()


class TestSettingsEndpoint:
    @pytest.mark.asyncio
    async def test_get_returns_every_default(self, db):
        """The console pre-fills and resets from these; it restates no prompt text."""
        body = (await _get(db)).json()
        assert body["caption_style_prompts"] == CAPTION_STYLES
        assert body["motion_style_prompts"] == MOTION_STYLE_PRESETS
        assert body["motion_template_default"] == MOTION_TEMPLATE
        assert set(body["motion_placeholders"]) >= {"{scene}", "{style}"}
        assert body["prompt_max_length"] == PROMPT_MAX_CHARS
        assert body["motion_instruction"] == "" and body["caption_instruction"] == ""

    @pytest.mark.asyncio
    async def test_an_unknown_placeholder_is_422(self, db):
        resp = await _put(db, {"motion_instruction": "Describe {sceen} {style}"})
        assert resp.status_code == 422
        assert "{sceen}" in resp.text
        assert await db.get(AppSetting, "motion_instruction") is None

    @pytest.mark.asyncio
    async def test_an_overlong_prompt_is_422(self, db):
        too_long = "x" * (PROMPT_MAX_CHARS + 1)
        assert (await _put(db, {"motion_instruction": too_long})).status_code == 422
        assert (await _put(db, {"caption_instruction": too_long})).status_code == 422

    @pytest.mark.asyncio
    async def test_a_valid_template_is_saved_as_written(self, db):
        t = "{#scene}Scene: {scene}\n\n{/scene}Slow motion. {style}"
        resp = await _put(db, {"motion_instruction": t})
        assert resp.status_code == 200
        assert resp.json()["motion_instruction"] == t

    @pytest.mark.asyncio
    async def test_a_legacy_override_is_still_accepted(self, db):
        resp = await _put(db, {"motion_instruction": "my own words"})
        assert resp.status_code == 200
        assert resp.json()["motion_instruction"] == "my own words"

    @pytest.mark.asyncio
    async def test_saving_the_default_text_stores_empty(self, db):
        """Saving the pre-filled editor untouched must not pin today's wording."""
        resp = await _put(db, {"motion_instruction": MOTION_TEMPLATE,
                               "caption_style": "rich",
                               "caption_instruction": CAPTION_STYLES["rich"]})
        assert resp.status_code == 200
        assert resp.json()["motion_instruction"] == ""
        assert resp.json()["caption_instruction"] == ""

    @pytest.mark.asyncio
    async def test_another_styles_text_is_a_real_override(self, db):
        """"rich" text under "standard" changes the prompt, so it must be kept."""
        resp = await _put(db, {"caption_style": "standard",
                               "caption_instruction": CAPTION_STYLES["rich"]})
        assert resp.json()["caption_instruction"] == CAPTION_STYLES["rich"]

    @pytest.mark.asyncio
    async def test_the_scene_endpoint_validates_the_same_way(self, db):
        client, app = await _client(db)
        try:
            async with client as c:
                resp = await c.post("/images/scene", params={"path": PATH},
                                    json={"motion_instruction": "{bogus}"})
        finally:
            app.dependency_overrides.clear()
        assert resp.status_code == 422


class TestTryEndpoint:
    async def _try(self, db, body=None, *, motion_enabled=True, fail=None):
        """POST /images/scene/try with S3 and the captioner stubbed at describe()."""
        from app.config import settings

        sent: list[str] = []

        async def fake_describe(image, instruction, base_url=None):
            sent.append(instruction)
            if fail:
                raise fail
            return "a woman on a sofa" if len(sent) == 1 else "she leans back slowly"

        async def fake_base(db, interactive):
            return "http://captioner"

        client, app = await _client(db)
        try:
            with patch("app.routes.images.download_bytes", return_value=b"png"), \
                 patch("app.routes.captions.describe", side_effect=fake_describe), \
                 patch("app.joycaption.describe", side_effect=fake_describe), \
                 patch("app.routes.captions._caption_base", side_effect=fake_base), \
                 patch.object(settings, "motion_caption_enabled", motion_enabled):
                async with client as c:
                    resp = await c.post("/images/scene/try", params={"path": PATH},
                                        json=body if body is not None else {})
        finally:
            app.dependency_overrides.clear()
        return resp, sent

    @pytest.mark.asyncio
    async def test_saved_prompts_when_nothing_is_sent(self, db):
        resp, sent = await self._try(db)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["caption"] == "a woman on a sofa"
        assert body["motion"] == "she leans back slowly"
        assert body["caption_instruction_used"] == CAPTION_STYLES["standard"]
        assert body["motion_instruction_used"] == motion_instruction_for(
            "handheld", "", "a woman on a sofa")
        assert sent == [body["caption_instruction_used"], body["motion_instruction_used"]]

    @pytest.mark.asyncio
    async def test_saved_overrides_are_used_when_nothing_is_sent(self, db):
        db.add(AppSetting(key="caption_instruction", value="Say what you see."))
        db.add(AppSetting(key="motion_instruction", value="Motion only. {style}"))
        db.add(AppSetting(key="motion_style", value="static"))
        await db.flush()
        body = (await self._try(db))[0].json()
        assert body["caption_instruction_used"] == "Say what you see."
        assert body["motion_instruction_used"] == ("Motion only. "
                                                   + MOTION_STYLE_PRESETS["static"]).strip()

    @pytest.mark.asyncio
    async def test_unsaved_text_is_used_and_nothing_is_stored(self, db):
        db.add(AppSetting(key="caption_instruction", value="saved caption prompt"))
        await db.flush()
        resp, sent = await self._try(db, {
            "caption_instruction": "Draft caption prompt.",
            "motion_template": "{#scene}Given: {scene}. {/scene}Draft motion. {style}",
            "motion_style": "none",
        })
        body = resp.json()
        assert body["caption_instruction_used"] == "Draft caption prompt."
        assert body["motion_instruction_used"] == "Given: a woman on a sofa. Draft motion."
        # Nothing stored: no image row, and the saved setting is untouched.
        assert await db.get(ImageMeta, PATH) is None
        assert (await db.get(AppSetting, "caption_instruction")).value == "saved caption prompt"
        assert await db.get(AppSetting, "motion_instruction") is None

    @pytest.mark.asyncio
    async def test_empty_means_the_default(self, db):
        db.add(AppSetting(key="caption_instruction", value="saved caption prompt"))
        db.add(AppSetting(key="motion_instruction", value="saved motion prompt"))
        await db.flush()
        body = (await self._try(db, {"caption_instruction": "", "caption_style": "terse",
                                     "motion_template": ""}))[0].json()
        assert body["caption_instruction_used"] == CAPTION_STYLES["terse"]
        assert body["motion_instruction_used"] == _pre_555("handheld", "", "a woman on a sofa")

    @pytest.mark.asyncio
    async def test_a_bad_template_is_422_before_the_captioner_is_called(self, db):
        resp, sent = await self._try(db, {"motion_template": "{nope}"})
        assert resp.status_code == 422
        assert sent == []

    @pytest.mark.asyncio
    async def test_motion_switched_off_is_reported(self, db):
        body = (await self._try(db, motion_enabled=False))[0].json()
        assert body["motion"] is None and body["motion_enabled"] is False
        assert body["motion_instruction_used"] is None

    @pytest.mark.asyncio
    async def test_a_captioner_failure_is_503(self, db):
        from app.joycaption import CaptionError
        resp, _ = await self._try(db, fail=CaptionError("down"))
        assert resp.status_code == 503

    @pytest.mark.asyncio
    async def test_it_takes_a_turn_in_the_caption_queue(self, db):
        """Tries must line up behind other captioning, not race it on the one-slot box."""
        from app.caption_queue import queue as caption_queue

        seen = []
        real_turn = caption_queue.turn

        def spy(path, **kw):
            seen.append((path, kw.get("kind")))
            return real_turn(path, **kw)

        with patch.object(caption_queue, "turn", side_effect=spy):
            resp, _ = await self._try(db)
        assert resp.status_code == 200
        # Labelled a try: it stores nothing, so the console must not show it as a caption of
        # the image on its way (console#564).
        assert seen == [(PATH, "try")]

    @pytest.mark.asyncio
    async def test_a_foreign_bucket_is_refused(self, db):
        client, app = await _client(db)
        try:
            async with client as c:
                resp = await c.post("/images/scene/try", params={"path": "s3://elsewhere/x.png"},
                                    json={})
        finally:
            app.dependency_overrides.clear()
        assert resp.status_code == 400
