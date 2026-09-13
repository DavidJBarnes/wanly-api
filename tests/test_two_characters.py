"""One character in one shot (wanly-console#473, post-#102).

The blob names the person as `characters: [{name, trigger, gender, char_lora, s1, s2}]`
(a list of ONE), with the older scalar keys mirrored from that entry. The two-person slot
(`<TRIGGER2>`, characters[1], the pre-#102 stacked-LoRA way) is gone: a two-person shot is
a JOINT character -- one LoRA trained on both identities, whose trigger phrase carries both
caption pairs -- plus scene text naming who is who.
"""
import pytest

from app.model_requirements import LORA, required_artifacts
from app.models import LtxCharacter
from app.recipe_blob import (
    character_phrase, placeholders_in, recipe_characters, render_prompt, trigger_phrase,
)
from app.routes.wildcards import RESERVED_WILDCARD_NAMES

ONE = {
    "recipe": "Missionary", "character": "p@y", "trigger": "p@y", "char_lora": "pay_v2_e05",
    "char_s1": 0.8, "char_s2": 1.5,
    "characters": [{"name": "p@y", "trigger": "p@y", "char_lora": "pay_v2_e05",
                    "s1": 0.8, "s2": 1.5}],
}
ONE_SCALAR = {"recipe": "Missionary", "character": "p@y", "trigger": "p@y",
              "char_lora": "pay_v2_e05", "char_s1": 0.8, "char_s2": 1.5}
#: A blob from the pre-joint era. The reader returns it as recorded; the RENDER uses only
#: the first entry.
TWO_LEGACY = {
    "recipe": "Bedroom, two", "character": "p@y", "trigger": "p@y", "char_lora": "pay_v2_e05",
    "characters": [
        {"name": "p@y", "trigger": "p@y", "char_lora": "pay_v2_e05", "s1": 0.8, "s2": 1.5},
        {"name": "Me", "trigger": "d@vid", "char_lora": "david_v1_final", "s1": 0.7, "s2": 1.2},
    ],
}


class TestRenderPrompt:
    def test_fills_the_one_placeholder(self):
        assert render_prompt("<TRIGGER>, a woman", "p@y") == "p@y, a woman"

    def test_a_list_takes_its_first_entry(self):
        """Callers built around the old two-slot shape still pass a list."""
        assert render_prompt("<TRIGGER> and more", ["p@y", "d@vid"]) == "p@y and more"

    def test_no_trigger_leaves_the_placeholder_literal_not_dropped(self):
        assert render_prompt("<TRIGGER>, a woman", None) == "<TRIGGER>, a woman"
        assert render_prompt("<TRIGGER>, a woman", "") == "<TRIGGER>, a woman"

    def test_a_template_without_the_placeholder_is_returned_unchanged(self):
        assert render_prompt("a woman standing", "p@y") == "a woman standing"

    def test_a_legacy_template_with_trigger2_only_fills_trigger(self):
        """A pre-joint template carrying <TRIGGER2> can no longer be produced (every pose
        was rewritten), but if one arrives the second placeholder is simply untouched."""
        assert render_prompt("<TRIGGER2> behind <TRIGGER>", "p@y") == "<TRIGGER2> behind p@y"


class TestTriggerPhrase:
    """What fills the placeholder is the caption the LoRA trained on, gender included
    (wanly-console#487): "p@yton, woman", not "p@yton"."""

    def test_the_phrase_is_the_training_caption(self):
        assert trigger_phrase("p@yton", "woman") == "p@yton, woman"
        assert trigger_phrase("d@vid", "man") == "d@vid, man"

    def test_no_gender_is_the_bare_trigger_as_before(self):
        assert trigger_phrase("k3llydw", None) == "k3llydw"
        assert trigger_phrase("k3llydw", "") == "k3llydw"

    def test_the_no_character_slot_never_grows_a_gender(self):
        assert trigger_phrase("", "woman") == ""
        assert trigger_phrase(None, "woman") is None

    def test_a_blob_entry_renders_its_recorded_gender(self):
        person = {"name": "Payton", "trigger": "p@yton", "gender": "woman"}
        assert character_phrase(person) == "p@yton, woman"
        assert character_phrase({"name": "Me", "trigger": "d@vid"}) == "d@vid"


class TestTheOneReader:
    def test_the_list_wins(self):
        assert [c["name"] for c in recipe_characters(ONE)] == ["p@y"]

    def test_a_legacy_two_entry_list_is_returned_as_recorded(self):
        assert [c["name"] for c in recipe_characters(TWO_LEGACY)] == ["p@y", "Me"]

    def test_the_scalar_shape_is_one_person(self):
        [one] = recipe_characters(ONE_SCALAR)
        assert one == {"name": "p@y", "trigger": "p@y", "char_lora": "pay_v2_e05",
                       "s1": 0.8, "s2": 1.5}

    def test_nothing_is_nothing(self):
        assert recipe_characters(None) == []
        assert recipe_characters({"recipe": "x"}) == []
        assert recipe_characters({"characters": []}) == []


class TestRequirements:
    def test_the_character_lora_is_a_requirement(self):
        names = {a.name for a in required_artifacts(ONE) if a.kind == LORA}
        assert names == {"pay_v2_e05"}

    def test_the_scalar_shape_still_yields_one(self):
        names = {a.name for a in required_artifacts(ONE_SCALAR) if a.kind == LORA}
        assert names == {"pay_v2_e05"}

    def test_a_none_slot_is_not_a_file(self):
        blob = dict(ONE, characters=[dict(ONE["characters"][0], char_lora="none")])
        assert {a.name for a in required_artifacts(blob) if a.kind == LORA} == set()


class TestTheGuard:
    def test_trigger_is_a_reserved_wildcard_name(self):
        assert "TRIGGER" in RESERVED_WILDCARD_NAMES

    def test_trigger2_stays_reserved_as_a_transition_guard(self):
        """The placeholder is gone, but the reservation stays one release so a stray
        template can never resolve <TRIGGER2> as a wildcard."""
        assert "TRIGGER2" in RESERVED_WILDCARD_NAMES

    def test_placeholders_in_sees_only_trigger(self):
        assert placeholders_in("<TRIGGER>, <SCENE>") == ["<TRIGGER>"]
        assert placeholders_in("no placeholders") == []


@pytest.mark.asyncio
class TestResolveTrigger:
    async def test_fills_the_one_placeholder(self, db):
        from app.routes.segments import _resolve_trigger
        db.add(LtxCharacter(name="p@y", char_lora="pay_v2_e05", trigger="p@y"))
        await db.commit()
        out = await _resolve_trigger(db, "<TRIGGER>, a woman", ONE)
        assert out == "p@y, a woman"

    async def test_the_rows_trigger_beats_the_recorded_one(self, db):
        """A trigger corrected on the character row applies to the next render."""
        from app.routes.segments import _resolve_trigger
        db.add(LtxCharacter(name="p@y", char_lora="pay_v2_e05", trigger="pay"))
        await db.commit()
        out = await _resolve_trigger(db, "<TRIGGER>, a woman", ONE)
        assert out == "pay, a woman"

    async def test_a_deleted_row_falls_back_to_what_was_recorded(self, db):
        from app.routes.segments import _resolve_trigger
        out = await _resolve_trigger(db, "<TRIGGER>, a woman", ONE)
        assert out == "p@y, a woman"

    async def test_a_scalar_blob_still_fills_one(self, db):
        from app.routes.segments import _resolve_trigger
        out = await _resolve_trigger(db, "<TRIGGER>, a woman", ONE_SCALAR)
        assert out == "p@y, a woman"

    async def test_the_rows_gender_renders_beside_the_trigger(self, db):
        """The caption was "p@y, woman"; the prompt says the same pair (wanly-console#487)."""
        from app.routes.segments import _resolve_trigger
        db.add(LtxCharacter(name="p@y", char_lora="pay_v2_e05", trigger="p@y", gender="woman"))
        await db.commit()
        out = await _resolve_trigger(db, "<TRIGGER>, kneeling", ONE)
        assert out == "p@y, woman, kneeling"

    async def test_a_joint_phrase_lands_whole_in_the_one_placeholder(self, db):
        """DavidPayton: one LoRA, both pairs in the trigger. Nothing splits it anymore."""
        from app.routes.segments import _resolve_trigger
        db.add(LtxCharacter(name="DavidPayton", char_lora="DavidPayton_v2_final",
                            trigger="p@yton, woman and d@vid, man"))
        await db.commit()
        blob = {"character": "DavidPayton", "trigger": "p@yton, woman and d@vid, man",
                "char_lora": "DavidPayton_v2_final",
                "characters": [{"name": "DavidPayton", "trigger": "p@yton, woman and d@vid, man",
                                "char_lora": "DavidPayton_v2_final"}]}
        out = await _resolve_trigger(db, "<TRIGGER>, a couch scene", blob)
        assert out == "p@yton, woman and d@vid, man, a couch scene"

    async def test_a_legacy_two_entry_blob_renders_only_the_first(self, db):
        """Pre-joint blobs name two people with two LoRAs; the second slot is gone, so
        only the first is rendered. The old renders are historical."""
        from app.routes.segments import _resolve_trigger
        db.add(LtxCharacter(name="p@y", char_lora="pay_v2_e05", trigger="p@y", gender="woman"))
        await db.commit()
        out = await _resolve_trigger(db, "<TRIGGER>, a scene", TWO_LEGACY)
        assert out == "p@y, woman, a scene"
