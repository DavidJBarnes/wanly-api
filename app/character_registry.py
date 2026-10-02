"""The character registry's rules, in one place (wanly-api#352).

A character is ONE PERSON = one trigger + one gender, owned by the registry. Before this the
training dialog took both as free text on every run, and David's face ended up with a
different trigger in each pair (`D@vidUdycz`, `D@vidK3lly-2000`, ...) -- each LoRA fine on
its own, and no two of them agreeing on what to call him.

Two readers need these rules: the character routes (which refuse to change a trained
character's trigger) and the training route (which captions with the registry's values and
publishes a pair's phrase). One module, so the two can never disagree about what "has
trained" or "the pair's phrase" means.
"""
from app.models import LtxCharacter


def has_trained(c: LtxCharacter) -> bool:
    """Has a LoRA been trained against this row's trigger and gender?

    Either sign is enough. `trained_from` is stamped by every publish; `char_lora` covers the
    rows that predate the stamp (and hand-registered LoRAs). "none" is the registry's value
    for "no LoRA yet" -- see LtxCharacterCreate -- and is not a LoRA.
    """
    if c.trained_from:
        return True
    lora = (c.char_lora or "").strip().lower()
    return bool(lora) and lora != "none"


def identity_phrase(trigger: str | None, gender: str | None) -> str | None:
    """One person's caption prefix: "d@vid, man". None without both halves.

    Both halves are required here, unlike recipe_blob.trigger_phrase (which renders a bare
    trigger for a character that predates genders): a TRAINING caption without the gender
    binds the face to nothing, which is exactly what the first run did.
    """
    if not trigger or not gender:
        return None
    return f"{trigger}, {gender}"


def pair_phrase(members: list[LtxCharacter]) -> str | None:
    """A pair's trigger: its members' phrases joined, in member order.

    "d@vid, man and k3lly2026, woman" -- exactly the prefix a composition caption carries,
    so the phrase that fills <TRIGGER> at render is the one the composition images taught.
    " and " because it is natural text inside one placeholder; the captions never held "&".
    None when any member lacks a trigger or a gender.
    """
    parts = [identity_phrase(m.trigger, m.gender) for m in members]
    if not parts or any(p is None for p in parts):
        return None
    return " and ".join(parts)


def _no_lora(char_lora: str | None) -> bool:
    lora = (char_lora or "").strip().lower()
    return not lora or lora == "none"


def is_draft(c: LtxCharacter, first_member: LtxCharacter | None = None) -> bool:
    """A DRAFT character has neither a LoRA nor an identity reference (wanly-console#592).

    Allowed since migration 110 so a brand-new, LoRA-less character can exist long enough for
    Build sheet to make its first sheet -- the sheet flow lives on an existing character. A
    draft cannot render: nothing in it carries the person, so the submit and the claim refuse
    it (see segments._draft_refusal) rather than render a stranger on the base model.

    A PAIR has no reference of its own; it renders with its FIRST member's. So a pair with no
    joint LoRA is a draft exactly when that member (passed in by the caller, which has the
    session) has no reference either.
    """
    if not _no_lora(c.char_lora):
        return False
    if (c.kind or "solo") == "pair":
        return first_member is None or not (first_member.sheet_uri or first_member.face_ref_uri)
    return not (c.sheet_uri or c.face_ref_uri)


def draft_message(name: str) -> str:
    """What a refused draft says, everywhere it is refused: one sentence, with the way out."""
    return f"{name} has no LoRA or sheet yet: build a sheet or attach a LoRA."
