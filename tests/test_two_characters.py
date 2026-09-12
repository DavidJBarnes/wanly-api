"""Two people in one shot (wanly-console#473).

The blob names them as `characters: [...]`, slot 0 fills <TRIGGER> and slot 1 <TRIGGER2>,
and the scalar keys stay mirrored from slot 0 so nothing written before the list is refused.
"""
import pytest

from app.model_requirements import LORA, required_artifacts
from app.models import LtxCharacter
from app.recipe_blob import (
    MAX_CHARACTERS, character_phrase, recipe_characters, recipe_problem,
    render_prompt, trigger_phrase,
)
from app.routes.wildcards import RESERVED_WILDCARD_NAMES

TWO = {
    "recipe": "Bedroom, two", "character": "p@y", "trigger": "p@y", "char_lora": "pay_v2_e05",
    "char_s1": 0.8, "char_s2": 1.5,
    "characters": [
        {"name": "p@y", "trigger": "p@y", "char_lora": "pay_v2_e05", "s1": 0.8, "s2": 1.5},
        {"name": "Me", "trigger": "d@vid", "char_lora": "david_v1_final", "s1": 0.7, "s2": 1.2},
    ],
}
ONE_SCALAR = {"recipe": "Missionary", "character": "p@y", "trigger": "p@y",
              "char_lora": "pay_v2_e05", "char_s1": 0.8, "char_s2": 1.5}


class TestRenderPrompt:
    def test_fills_both_slots_in_order(self):
        assert render_prompt("<TRIGGER2> behind <TRIGGER>", ["p@y", "d@vid"]) == "d@vid behind p@y"

    def test_a_missing_second_trigger_is_left_literal_not_dropped(self):
        assert render_prompt("<TRIGGER> and <TRIGGER2>", ["p@y"]) == "p@y and <TRIGGER2>"
        assert render_prompt("<TRIGGER> and <TRIGGER2>", ["p@y", None]) == "p@y and <TRIGGER2>"

    def test_still_takes_a_single_string(self):
        assert render_prompt("<TRIGGER>, a woman", "p@y") == "p@y, a woman"

    def test_trigger_does_not_eat_trigger2(self):
        """"<TRIGGER>" is not a substring of "<TRIGGER2>", and this proves it stays so."""
        assert render_prompt("<TRIGGER2>", ["p@y"]) == "<TRIGGER2>"


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

    def test_both_people_render_bound_to_their_gender(self):
        people = [{"trigger": "p@yton", "gender": "woman"}, {"trigger": "d@vid", "gender": "man"}]
        assert render_prompt("<TRIGGER> and <TRIGGER2>", [character_phrase(p) for p in people]) \
            == "p@yton, woman and d@vid, man"


class TestTheOneReader:
    def test_the_list_wins(self):
        assert [c["name"] for c in recipe_characters(TWO)] == ["p@y", "Me"]

    def test_the_scalar_shape_is_one_person(self):
        [one] = recipe_characters(ONE_SCALAR)
        assert one == {"name": "p@y", "trigger": "p@y", "char_lora": "pay_v2_e05",
                       "s1": 0.8, "s2": 1.5}

    def test_nothing_is_nothing(self):
        assert recipe_characters(None) == []
        assert recipe_characters({"recipe": "x"}) == []
        assert recipe_characters({"characters": []}) == []


class TestRequirements:
    def test_every_character_lora_is_a_requirement(self):
        names = {a.name for a in required_artifacts(TWO) if a.kind == LORA}
        assert {"pay_v2_e05", "david_v1_final"} <= names

    def test_the_scalar_shape_still_yields_one(self):
        names = {a.name for a in required_artifacts(ONE_SCALAR) if a.kind == LORA}
        assert names == {"pay_v2_e05"}

    def test_a_none_slot_is_not_a_file(self):
        blob = dict(TWO, characters=[TWO["characters"][0], {"name": "x", "char_lora": "none"}])
        names = {a.name for a in required_artifacts(blob) if a.kind == LORA}
        assert names == {"pay_v2_e05"}


class TestValidation:
    def test_a_third_character_is_refused(self):
        blob = dict(TWO, characters=TWO["characters"] + [{"name": "third", "char_lora": "t"}])
        assert "at most 2" in recipe_problem(blob, "<TRIGGER>")
        assert MAX_CHARACTERS == 2

    def test_trigger2_in_the_prompt_needs_a_second_character(self):
        assert "second person" in recipe_problem(ONE_SCALAR, "<TRIGGER> with <TRIGGER2>")

    def test_a_two_person_blob_with_a_two_person_prompt_is_fine(self):
        assert recipe_problem(TWO, "<TRIGGER> with <TRIGGER2>") is None

    def test_old_blobs_are_never_refused(self):
        assert recipe_problem(ONE_SCALAR, "p@y, a woman") is None
        assert recipe_problem(None, "<TRIGGER2>") is None

    def test_the_routes_check_before_resolving(self):
        import inspect
        from app.routes import jobs, segments
        assert "recipe_problem(seg.ltx_recipe, seg.prompt)" in inspect.getsource(jobs.create_job)
        src = inspect.getsource(segments)
        assert src.index("recipe_problem(body.ltx_recipe, body.prompt)") < src.index(
            "prompt = await _resolve_trigger(db, body.prompt, body.ltx_recipe)")

    def test_trigger2_is_a_reserved_wildcard_name(self):
        assert "TRIGGER2" in RESERVED_WILDCARD_NAMES


@pytest.mark.asyncio
class TestResolveTrigger:
    async def test_fills_trigger2_from_the_second_character(self, db):
        from app.routes.segments import _resolve_trigger
        db.add(LtxCharacter(name="p@y", char_lora="pay_v2_e05", trigger="p@y"))
        db.add(LtxCharacter(name="Me", char_lora="david_v1_final", trigger="d@vid"))
        await db.commit()
        out = await _resolve_trigger(db, "<TRIGGER2> stands behind <TRIGGER>", TWO)
        assert out == "d@vid stands behind p@y"

    async def test_the_rows_trigger_beats_the_recorded_one(self, db):
        """A trigger corrected on the character row applies to the next render."""
        from app.routes.segments import _resolve_trigger
        db.add(LtxCharacter(name="Me", char_lora="david_v1_final", trigger="dav1d"))
        await db.commit()
        out = await _resolve_trigger(db, "<TRIGGER> and <TRIGGER2>", TWO)
        assert out.endswith("and dav1d")

    async def test_a_deleted_row_falls_back_to_what_was_recorded(self, db):
        from app.routes.segments import _resolve_trigger
        out = await _resolve_trigger(db, "<TRIGGER> and <TRIGGER2>", TWO)
        assert out == "p@y and d@vid"

    async def test_a_scalar_blob_still_fills_one(self, db):
        from app.routes.segments import _resolve_trigger
        out = await _resolve_trigger(db, "<TRIGGER>, a woman", ONE_SCALAR)
        assert out == "p@y, a woman"

    async def test_the_rows_gender_renders_beside_the_trigger(self, db):
        """The caption was "p@y, woman" / "d@vid, man"; the prompt says the same pair
        (wanly-console#487)."""
        from app.routes.segments import _resolve_trigger
        db.add(LtxCharacter(name="p@y", char_lora="pay_v2_e05", trigger="p@y", gender="woman"))
        db.add(LtxCharacter(name="Me", char_lora="david_v1_final", trigger="d@vid", gender="man"))
        await db.commit()
        out = await _resolve_trigger(db, "<TRIGGER2> stands behind <TRIGGER>", TWO)
        assert out == "d@vid, man stands behind p@y, woman"

    async def test_a_deleted_row_renders_the_gender_the_blob_recorded(self, db):
        from app.routes.segments import _resolve_trigger
        blob = {**TWO, "characters": [
            {**TWO["characters"][0], "gender": "woman"},
            {**TWO["characters"][1], "gender": "man"},
        ]}
        out = await _resolve_trigger(db, "<TRIGGER> and <TRIGGER2>", blob)
        assert out == "p@y, woman and d@vid, man"


class TestJointPhrase:
    """A JOINT character (wanly-api#102) carries BOTH identities in ONE trigger, joined by
    " and ". The render splits it back apart so each pair fills its own placeholder.

    The bug this fixes: the whole joint phrase landed in <TRIGGER> and <TRIGGER2> was left
    empty, so a two-person pose read "p@yton, woman and d@vid, man and . <scene>" -- both
    triggers dumped in one place instead of each beside its person, and the model favoured
    the first (Payton held, David drifted)."""

    JOINT = "p@yton, woman and d@vid, man"

    def test_a_two_person_pose_splits_each_pair_to_its_own_placeholder(self):
        out = render_prompt("<TRIGGER> and <TRIGGER2>, a scene", [self.JOINT])
        assert out == "p@yton, woman and d@vid, man, a scene"

    def test_the_split_places_each_trigger_beside_its_person(self):
        out = render_prompt("<TRIGGER> grips <TRIGGER2>", [self.JOINT])
        assert out == "p@yton, woman grips d@vid, man"

    def test_a_one_person_pose_keeps_the_whole_phrase(self):
        """Splitting here would leave the second identity nowhere to go -- DROPPING it."""
        out = render_prompt("<TRIGGER>, a scene", [self.JOINT])
        assert out == "p@yton, woman and d@vid, man, a scene"

    def test_the_legacy_ampersand_phrase_still_splits(self):
        out = render_prompt("<TRIGGER> and <TRIGGER2>", ["p@yton, woman & d@vid, man"])
        assert out == "p@yton, woman and d@vid, man"

    def test_a_single_phrase_is_untouched(self):
        assert render_prompt("<TRIGGER> and <TRIGGER2>", ["p@yton, woman"]) == \
            "p@yton, woman and <TRIGGER2>"

    def test_a_trigger_containing_and_is_not_mangled(self):
        """" and " in a normal phrase must not split unless it yields two real parts."""
        assert render_prompt("<TRIGGER> and <TRIGGER2>", ["rock and roll"]) == \
            "rock and roll and <TRIGGER2>"
