"""The recipe blob's people, read one way everywhere (wanly-console#473).

A segment's `ltx_recipe` names the people in the shot as

    characters: [{name, trigger, gender, char_lora, s1, s2}, ...]   # ordered, at most two

with the older scalar keys -- `character`, `trigger`, `char_lora`, `char_s1`, `char_s2` --
mirrored from the first entry so that everything written before the list existed, and
everything that has not learned about it, keeps working. Slot 0 fills `<TRIGGER>` in a
pose's prompt, slot 1 fills `<TRIGGER2>`; a pose is a two-person pose exactly when its
template uses `<TRIGGER2>`.

This module is the one reader of that shape. Three places used to each read the scalars on
their own (the trigger fill, the claim's requirements, the console's mirror of both); a
second person would have meant three copies of the list-or-scalar rule.
"""
from typing import Any, Sequence

TRIGGER_PLACEHOLDER = "<TRIGGER>"
TRIGGER2_PLACEHOLDER = "<TRIGGER2>"
#: In slot order. Index i is filled by characters[i].
TRIGGER_PLACEHOLDERS = (TRIGGER_PLACEHOLDER, TRIGGER2_PLACEHOLDER)
MAX_CHARACTERS = len(TRIGGER_PLACEHOLDERS)


def recipe_characters(ltx_recipe: dict[str, Any] | None) -> list[dict[str, Any]]:
    """The people in a recipe blob, `[{name, trigger, gender, char_lora, s1, s2}, ...]`.

    `gender` is recorded since wanly-console#487 and absent before; readers use `.get`.

    The list when it is there, else one entry synthesised from the scalars, else nothing.
    Entries are returned as recorded -- including a `char_lora` of "none", which is a real
    choice (render this slot on the base model) that the callers filter for themselves.
    """
    if not ltx_recipe:
        return []
    raw = ltx_recipe.get("characters")
    if isinstance(raw, list) and raw:
        return [dict(e) for e in raw if isinstance(e, dict)]
    if not any(ltx_recipe.get(k) for k in ("character", "char_lora")):
        return []
    return [{
        "name": ltx_recipe.get("character"),
        "trigger": ltx_recipe.get("trigger"),
        "char_lora": ltx_recipe.get("char_lora"),
        "s1": ltx_recipe.get("char_s1"),
        "s2": ltx_recipe.get("char_s2"),
    }]


def trigger_phrase(trigger: str | None, gender: str | None) -> str | None:
    """What fills a placeholder: the trigger AND the word its LoRA bound it to.

    Every run captions its images "<trigger>, <gender>" (wanly-api#293), so "p@yton, woman"
    is the token pair the identity actually learned, and the render prompt has to say the
    same thing. With two identity LoRAs summed into the same weights this pair is the only
    thing that says which face goes on which body (wanly-console#487).

    No trigger means no phrase -- the "no character" slot never grows a gender -- and no
    gender means the bare trigger, which is exactly what every character rendered before.
    """
    if not trigger:
        return trigger
    if not gender:
        return trigger
    return f"{trigger}, {gender}"


def character_phrase(person: dict[str, Any]) -> str | None:
    """`trigger_phrase` for one entry of `recipe_characters`."""
    return trigger_phrase(person.get("trigger"), person.get("gender"))


def render_prompt(template: str, triggers: str | Sequence[str | None]) -> str:
    """Fill a pose's placeholders with the characters' trigger words, slot by slot.

    A single string is the one-person shorthand. A slot with no trigger leaves its
    placeholder in place: rendering the literal text is bad, but silently dropping the token
    that anchors a character LoRA is worse and much harder to notice. A template with no
    placeholder at all is returned unchanged rather than rejected.
    """
    if isinstance(triggers, str):
        triggers = [triggers]
    out = template
    for placeholder, trigger in zip(TRIGGER_PLACEHOLDERS, triggers):
        if trigger:
            out = out.replace(placeholder, trigger)
    return out


def placeholders_in(text: str) -> list[str]:
    return [p for p in TRIGGER_PLACEHOLDERS if p in (text or "")]


def recipe_problem(ltx_recipe: dict[str, Any] | None, prompt: str | None) -> str | None:
    """Why this blob cannot render, or None.

    Light on purpose: the blob stays an untyped record so that nothing written before this
    is ever refused. Only the two things that would otherwise fail ten minutes into a
    claimed segment are caught here -- more people than the engine has room for, and a
    prompt still naming a second person the blob does not carry.
    """
    if not ltx_recipe:
        return None
    people = recipe_characters(ltx_recipe)
    if len(people) > MAX_CHARACTERS:
        return f"{len(people)} characters; a render takes at most {MAX_CHARACTERS}"
    if TRIGGER2_PLACEHOLDER in (prompt or "") and len(people) < 2:
        return (f"the prompt names a second person ({TRIGGER2_PLACEHOLDER}) but the recipe "
                f"carries {len(people)} character{'' if len(people) == 1 else 's'}")
    return None
