"""The caption and motion prompts are editable from Settings (console#555, console#573).

The promises, each pinned here:

1. NOBODY WHO DOES NOT TOUCH THE EDITORS SEES A DIFFERENT PROMPT. The motion prompt was a
   hand-assembled string, then a template (#555), and is now plain instructions the API wraps
   in the style sentence and grounding (#573). Each step must give byte-for-byte what the
   hand-assembled prompt gave, with a scene and without one, for every capture style. The
   findings in app/joycaption.py above MOTION_BASE are why the exact words matter.
2. A SAVED #555 TEMPLATE BECOMES ITS INSTRUCTIONS: tags and grounding sections stripped,
   since the API now adds exactly those.
3. AN OVERRIDE WRITTEN BEFORE #555 KEEPS ITS MEANING: the whole prompt, no style, no grounding.
4. "Try" stores nothing, and shows the full text sent.
"""
from unittest.mock import patch

import pytest

from app.joycaption import (CAPTION_STYLES, MOTION_BASE, MOTION_DEFAULT_STYLE,
                            MOTION_GROUNDING, MOTION_INSTRUCTIONS, MOTION_STYLE_PRESETS,
                            MOTION_TAIL, PROMPT_MAX_CHARS, compose_motion_prompt,
                            has_motion_tags, motion_instruction_for, resolve_motion_override,
                            strip_motion_tags)
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


#: The #555 default template, copied verbatim -- the value a console that saved its
#: pre-filled editor before the API normalised it would have stored.
TEMPLATE_555 = (
    "{#scene}Scene: {scene}\n\n{/scene}"
    + MOTION_BASE + "{style}" + MOTION_TAIL
    + "{#scene}\n\n" + MOTION_GROUNDING
    + " Do not restate the scene description.{/scene}"
)

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
        """Today a whitespace scene dropped the grounding; so must the instructions."""
        assert motion_instruction_for("handheld", "", scene) == _pre_555("handheld", "", scene)
        assert "Scene:" not in motion_instruction_for("handheld", "", scene)

    def test_a_scene_with_padding_is_stripped(self):
        assert (motion_instruction_for("amateur", "", f"  {SCENE}\n")
                == _pre_555("amateur", "", f"  {SCENE}\n"))

    @pytest.mark.parametrize("style", ALL_STYLES)
    def test_the_default_instructions_composed_directly(self, style):
        """The instructions themselves, not just motion_instruction_for, are what the console
        pre-fills and a user saves -- so saving them untouched must also be a no-op."""
        sentence = MOTION_STYLE_PRESETS.get(style, MOTION_STYLE_PRESETS["handheld"])
        assert compose_motion_prompt(MOTION_INSTRUCTIONS, sentence, SCENE) == \
            _pre_555(style, "", SCENE)
        assert compose_motion_prompt(MOTION_INSTRUCTIONS, sentence) == _pre_555(style)
        # And a saved copy of the default goes down the custom path with the same result.
        assert motion_instruction_for(style, MOTION_INSTRUCTIONS, SCENE) == \
            _pre_555(style, "", SCENE)

    def test_the_editor_text_has_no_tags(self):
        assert not has_motion_tags(MOTION_INSTRUCTIONS)
        assert "{" not in MOTION_INSTRUCTIONS and "Scene:" not in MOTION_INSTRUCTIONS
        assert MOTION_GROUNDING not in MOTION_INSTRUCTIONS

    def test_it_fits_under_the_cap(self):
        assert len(MOTION_INSTRUCTIONS) < PROMPT_MAX_CHARS
        assert all(len(v) < PROMPT_MAX_CHARS for v in CAPTION_STYLES.values())


class TestComposition:
    def test_the_style_goes_before_the_anchor_sentence(self):
        out = compose_motion_prompt("Move. Single continuous shot, no cuts.", "Handheld. ")
        assert out == "Move. Handheld.  Single continuous shot, no cuts."

    def test_without_the_anchor_the_style_goes_at_the_end(self):
        assert compose_motion_prompt("She leans back.", "Handheld. ") == \
            "She leans back. Handheld."
        assert compose_motion_prompt("She leans back.", MOTION_STYLE_PRESETS["none"]) == \
            "She leans back."

    def test_the_grounding_is_added_around_edited_instructions(self):
        out = motion_instruction_for("static", "She leans back slowly.", SCENE)
        assert out == (f"Scene: {SCENE}\n\nShe leans back slowly. "
                       f"{MOTION_STYLE_PRESETS['static'].strip()}\n\n{MOTION_GROUNDING} "
                       "Do not restate the scene description.")

    def test_no_scene_no_grounding(self):
        out = motion_instruction_for("none", "She leans back slowly.")
        assert out == "She leans back slowly."

    def test_braces_are_just_text(self):
        """Plain instructions have no syntax: a brace that is not a #555 tag is prose."""
        text = "Use JSON-ish {a b} notes, a {sceen} typo and a lone { brace."
        assert strip_motion_tags(text) == text
        assert motion_instruction_for("none", text) == text

    def test_a_scene_containing_tag_text_is_inserted_as_text(self):
        out = motion_instruction_for("none", "Move.", "she says {style}")
        assert out.startswith("Scene: she says {style}\n\nMove.")


class TestMigratingA555Template:
    def test_the_555_default_becomes_the_default_instructions(self):
        assert strip_motion_tags(TEMPLATE_555) == MOTION_INSTRUCTIONS
        assert resolve_motion_override("", TEMPLATE_555) == ("", "")

    @pytest.mark.parametrize("style", ALL_STYLES)
    def test_an_edited_555_template_keeps_its_words_and_its_grounding(self, style):
        """An edit of the #555 default renders exactly as it did under the template."""
        edited = TEMPLATE_555.replace("under 110 words", "under 80 words")
        instructions, legacy = resolve_motion_override("", edited)
        assert legacy == "" and "under 80 words" in instructions
        assert not has_motion_tags(instructions)
        expected = _pre_555(style, "", SCENE).replace("under 110 words", "under 80 words")
        assert motion_instruction_for(style, instructions, SCENE) == expected

    def test_text_before_and_after_the_sections_is_kept_and_sections_go(self):
        t = "{#scene}Given: {scene}. {/scene}Draft motion. {style}{#scene} Keep faces.{/scene}"
        assert strip_motion_tags(t) == "Draft motion."

    def test_a_bare_scene_tag_is_dropped(self):
        assert strip_motion_tags("Describe {scene} in motion. {style}") == \
            "Describe in motion."

    def test_stripping_is_idempotent(self):
        once = strip_motion_tags(TEMPLATE_555)
        assert strip_motion_tags(once) == once

    def test_a_template_passed_as_instructions_is_read_the_same_way(self):
        """A console tab opened before #573 still sends templates."""
        assert motion_instruction_for("handheld", TEMPLATE_555, SCENE) == \
            _pre_555("handheld", "", SCENE)


class TestLegacyWholePromptOverrides:
    """A custom motion instruction saved before #555 has no tags, in the old key."""

    def test_it_is_still_the_whole_prompt(self):
        legacy = "Describe the motion in this frame as a ten second clip."
        assert resolve_motion_override("", legacy) == ("", legacy)
        assert motion_instruction_for("handheld", "", SCENE, legacy) == legacy
        assert motion_instruction_for("handheld", "", SCENE, legacy) == \
            _pre_555("handheld", legacy, SCENE)

    def test_no_style_or_grounding_is_added(self):
        out = motion_instruction_for("handheld", "", SCENE, "  my own words  ")
        assert out == "my own words"
        assert SCENE not in out and MOTION_GROUNDING not in out

    def test_instructions_saved_since_win_over_it(self):
        assert resolve_motion_override("New words.", "old whole prompt") == ("New words.", "")
        assert motion_instruction_for("none", "New words.", "", "old whole prompt") == \
            "New words."


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
        assert body["motion_instruction_default"] == MOTION_INSTRUCTIONS
        assert "motion_placeholders" not in body and "motion_template_default" not in body
        assert body["prompt_max_length"] == PROMPT_MAX_CHARS
        assert body["motion_instruction"] == "" and body["caption_instruction"] == ""
        assert body["motion_legacy_prompt"] == ""

    @pytest.mark.asyncio
    async def test_an_overlong_prompt_is_422(self, db):
        too_long = "x" * (PROMPT_MAX_CHARS + 1)
        assert (await _put(db, {"motion_instruction": too_long})).status_code == 422
        assert (await _put(db, {"caption_instruction": too_long})).status_code == 422

    @pytest.mark.asyncio
    async def test_plain_instructions_are_saved_as_written(self, db):
        resp = await _put(db, {"motion_instruction": "Slow motion, {a b} and all."})
        assert resp.status_code == 200
        assert resp.json()["motion_instruction"] == "Slow motion, {a b} and all."
        assert (await db.get(AppSetting, "motion_prompt_instructions")).value == \
            "Slow motion, {a b} and all."

    @pytest.mark.asyncio
    async def test_a_template_sent_by_a_stale_console_is_stored_stripped(self, db):
        t = "{#scene}Scene: {scene}\n\n{/scene}Slow motion. {style}"
        resp = await _put(db, {"motion_instruction": t})
        assert resp.status_code == 200
        assert resp.json()["motion_instruction"] == "Slow motion."

    @pytest.mark.asyncio
    async def test_saving_the_default_text_stores_empty(self, db):
        """Saving the pre-filled editor untouched must not pin today's wording."""
        resp = await _put(db, {"motion_instruction": MOTION_INSTRUCTIONS,
                               "caption_style": "rich",
                               "caption_instruction": CAPTION_STYLES["rich"]})
        assert resp.status_code == 200
        assert resp.json()["motion_instruction"] == ""
        assert resp.json()["caption_instruction"] == ""
        # And so does the #555 default template, from a tab opened before #573.
        resp = await _put(db, {"motion_instruction": TEMPLATE_555})
        assert resp.json()["motion_instruction"] == ""

    @pytest.mark.asyncio
    async def test_another_styles_text_is_a_real_override(self, db):
        """"rich" text under "standard" changes the prompt, so it must be kept."""
        resp = await _put(db, {"caption_style": "standard",
                               "caption_instruction": CAPTION_STYLES["rich"]})
        assert resp.json()["caption_instruction"] == CAPTION_STYLES["rich"]

    @pytest.mark.asyncio
    async def test_a_saved_555_template_is_read_as_its_instructions(self, db):
        db.add(AppSetting(key="motion_instruction",
                          value="{#scene}Scene: {scene}\n\n{/scene}Slow. {style}"))
        await db.flush()
        body = (await _get(db)).json()
        assert body["motion_instruction"] == "Slow."
        assert body["motion_legacy_prompt"] == ""

    @pytest.mark.asyncio
    async def test_a_saved_555_default_is_read_as_the_default(self, db):
        db.add(AppSetting(key="motion_instruction", value=TEMPLATE_555))
        await db.flush()
        body = (await _get(db)).json()
        assert body["motion_instruction"] == "" and body["motion_legacy_prompt"] == ""

    @pytest.mark.asyncio
    async def test_a_legacy_whole_prompt_is_reported_and_kept(self, db):
        db.add(AppSetting(key="motion_instruction", value="my own words"))
        await db.flush()
        body = (await _get(db)).json()
        assert body["motion_instruction"] == ""
        assert body["motion_legacy_prompt"] == "my own words"
        # A save that does not touch the motion prompt leaves it in force.
        body = (await _put(db, {"motion_style": "static"})).json()
        assert body["motion_legacy_prompt"] == "my own words"

    @pytest.mark.asyncio
    async def test_saving_instructions_replaces_a_legacy_override(self, db):
        db.add(AppSetting(key="motion_instruction", value="my own words"))
        await db.flush()
        body = (await _put(db, {"motion_instruction": "New words."})).json()
        assert body["motion_instruction"] == "New words."
        assert body["motion_legacy_prompt"] == ""
        assert (await db.get(AppSetting, "motion_instruction")).value == ""

    @pytest.mark.asyncio
    async def test_resetting_to_default_replaces_a_legacy_override(self, db):
        db.add(AppSetting(key="motion_instruction", value="my own words"))
        await db.flush()
        body = (await _put(db, {"motion_instruction": ""})).json()
        assert body["motion_instruction"] == "" and body["motion_legacy_prompt"] == ""


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
        db.add(AppSetting(key="motion_prompt_instructions", value="Motion only."))
        db.add(AppSetting(key="motion_style", value="static"))
        await db.flush()
        body = (await self._try(db))[0].json()
        assert body["caption_instruction_used"] == "Say what you see."
        # The full text sent: the instructions with the style and grounding the API adds.
        assert body["motion_instruction_used"] == motion_instruction_for(
            "static", "Motion only.", "a woman on a sofa")
        assert body["motion_instruction_used"].startswith("Scene: a woman on a sofa\n\n")
        assert MOTION_GROUNDING in body["motion_instruction_used"]

    @pytest.mark.asyncio
    async def test_a_saved_legacy_whole_prompt_is_sent_verbatim(self, db):
        db.add(AppSetting(key="motion_instruction", value="Whole prompt of my own."))
        await db.flush()
        body = (await self._try(db))[0].json()
        assert body["motion_instruction_used"] == "Whole prompt of my own."

    @pytest.mark.asyncio
    async def test_the_old_field_name_still_works(self, db):
        """A Settings tab opened before #573 sends motion_template, and a template in it."""
        body = (await self._try(db, {
            "motion_template": "{#scene}Scene: {scene}\n\n{/scene}Old tab. {style}",
            "motion_style": "none"}))[0].json()
        assert body["motion_instruction_used"] == motion_instruction_for(
            "none", "Old tab.", "a woman on a sofa")

    @pytest.mark.asyncio
    async def test_unsaved_text_is_used_and_nothing_is_stored(self, db):
        db.add(AppSetting(key="caption_instruction", value="saved caption prompt"))
        await db.flush()
        resp, sent = await self._try(db, {
            "caption_instruction": "Draft caption prompt.",
            "motion_instruction": "Draft motion.",
            "motion_style": "none",
        })
        body = resp.json()
        assert body["caption_instruction_used"] == "Draft caption prompt."
        assert body["motion_instruction_used"] == (
            f"Scene: a woman on a sofa\n\nDraft motion.\n\n{MOTION_GROUNDING} "
            "Do not restate the scene description.")
        # Nothing stored: no image row, and the saved setting is untouched.
        assert await db.get(ImageMeta, PATH) is None
        assert (await db.get(AppSetting, "caption_instruction")).value == "saved caption prompt"
        assert await db.get(AppSetting, "motion_prompt_instructions") is None

    @pytest.mark.asyncio
    async def test_empty_means_the_default(self, db):
        db.add(AppSetting(key="caption_instruction", value="saved caption prompt"))
        # A legacy whole prompt too: "" is the default, not "whatever is saved".
        db.add(AppSetting(key="motion_instruction", value="saved motion prompt"))
        await db.flush()
        body = (await self._try(db, {"caption_instruction": "", "caption_style": "terse",
                                     "motion_instruction": ""}))[0].json()
        assert body["caption_instruction_used"] == CAPTION_STYLES["terse"]
        assert body["motion_instruction_used"] == _pre_555("handheld", "", "a woman on a sofa")

    @pytest.mark.asyncio
    async def test_an_overlong_prompt_is_422_before_the_captioner_is_called(self, db):
        resp, sent = await self._try(db, {"motion_instruction": "x" * (PROMPT_MAX_CHARS + 1)})
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

        def spy(path):
            seen.append(path)
            return real_turn(path)

        with patch.object(caption_queue, "turn", side_effect=spy):
            resp, _ = await self._try(db)
        assert resp.status_code == 200
        assert seen == [PATH]

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
