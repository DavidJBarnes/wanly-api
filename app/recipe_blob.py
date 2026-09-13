"""The recipe blob's person, read one way everywhere (wanly-console#473).

A segment's `ltx_recipe` names the person in the shot as

    characters: [{name, trigger, gender, char_lora, s1, s2}]   # ONE entry

with the older scalar keys -- `character`, `trigger`, `char_lora`, `char_s1`, `char_s2` --
mirrored from that entry so that everything written before the list existed, and everything
that has not learned about it, keeps working.

ONE person per render, always. The two-person slot (<TRIGGER2> / characters[1], the
pre-#102 way of putting two people in a shot with two stacked LoRAs) was removed along with
its placeholder: a two-person shot is now a JOINT character -- one LoRA trained on both
identities, whose trigger phrase carries both caption pairs -- plus scene text naming who is
who. The list shape stays (a list of one) because the blob is a record and older readers
still expect the list.

This module is the one reader of that shape. Three places used to each read the scalars on
their own (the trigger fill, the claim's requirements, the console's mirror of both); a
second reader would have meant three copies of the list-or-scalar rule.
"""
from typing import Any

TRIGGER_PLACEHOLDER = "<TRIGGER>"
#: One placeholder. A tuple so the zip-based fill and `placeholders_in` keep their shape.
TRIGGER_PLACEHOLDERS = (TRIGGER_PLACEHOLDER,)


def recipe_characters(ltx_recipe: dict[str, Any] | None) -> list[dict[str, Any]]:
    """The person in a recipe blob, `[{name, trigger, gender, char_lora, s1, s2}]`.

    `gender` is recorded since wanly-console#487 and absent before; readers use `.get`.

    The list when it is there, else one entry synthesised from the scalars, else nothing.
    Entries are returned as recorded -- including a `char_lora` of "none", which is a real
    choice (render on the base model) that the callers filter for themselves. A legacy blob
    may carry two entries from the pre-joint era; only the first is rendered.
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
    same thing. A JOINT character's trigger already carries every pair ("p@yton, woman and
    d@vid, man") and renders whole -- nothing splits it anymore.

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


def render_prompt(template: str, triggers: str | Any) -> str:
    """Fill a pose's <TRIGGER> with the character's trigger phrase.

    A single string is the shorthand; a list is accepted for callers built around the old
    two-slot shape, and its first entry wins. No trigger leaves the placeholder in place:
    rendering the literal text is bad, but silently dropping the token that anchors a
    character LoRA is worse and much harder to notice. A template with no placeholder at all
    is returned unchanged rather than rejected.

    The JOINT character's whole phrase lands in the one placeholder, and the scene text
    names who is who -- that is the post-#102 design, after <TRIGGER2> was removed.
    """
    first = triggers[0] if isinstance(triggers, (list, tuple)) else triggers
    if first:
        template = template.replace(TRIGGER_PLACEHOLDER, first)
    return template


def placeholders_in(text: str) -> list[str]:
    return [p for p in TRIGGER_PLACEHOLDERS if p in (text or "")]
