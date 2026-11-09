"""Prompts for character face loops (wanly-api#495, wanly-console#742).

Each preset is the MOTION part of a calm portrait prompt. The wording is what produced the loops
David judged "very good" on the mannequin (Kelly-2026 jobs 465275c6, 185c719e): an LTX render on
the Prompt recipe, from a close-up photo, with the head held still. Stillness is the point — on
a static mannequin any head or jaw motion reads as the whole face sliding — so every preset ends
with the same locked-camera sentence, and the negative prompt names the motions to avoid.
"""
from __future__ import annotations

STILL = ("Her head stays perfectly still and centered in frame. Static camera on a tripod, locked off, "
         "no camera movement, no zoom, no cuts.")
NEG_CLOSED = "talking, laughing, open mouth, teeth, head turning, head tilting, nodding, camera movement, zoom, cuts, blur"
NEG_SMILE = "laughing, talking, wide open mouth, head turning, head tilting, nodding, camera movement, zoom, cuts, blur"

PRESETS: dict[str, dict[str, str]] = {
    "calm": {
        "label": "Calm smile + blinks",
        "motion": "She blinks naturally every couple of seconds, and her lips slowly curl into a small "
                  "closed-mouth smile, then relax. Calm, subtle expression.",
        "negative": NEG_CLOSED,
    },
    "neutral": {
        "label": "Neutral + blinks",
        "motion": "She holds a calm, neutral expression with her mouth closed and blinks slowly every couple of seconds.",
        "negative": NEG_CLOSED,
    },
    "brows": {
        "label": "Eyebrow raise",
        "motion": "She holds a calm, neutral expression with her mouth closed, blinks slowly twice, and briefly "
                  "raises her eyebrows a little before relaxing.",
        "negative": NEG_CLOSED,
    },
    "smile": {
        "label": "Warm smile",
        "motion": "She smiles warmly at the camera with a happy, genuine smile showing her teeth, and holds the "
                  "smile while blinking naturally. Her mouth stays still in the smile.",
        "negative": NEG_SMILE,
    },
    "smile_grow": {
        "label": "Smile grows",
        "motion": "She starts with a calm, neutral expression, then her face slowly breaks into a big warm smile "
                  "showing her teeth, holds it for a moment, blinks, and gently relaxes.",
        "negative": NEG_SMILE,
    },
    "custom": {"label": "Custom", "motion": "", "negative": NEG_SMILE},
}

FRAMES = 121  # 5 s at 24 fps: plenty for a 3 s window, half the GPU time of a 241-frame render
FPS = 24


def pronoun(gender: str | None) -> str:
    return {"woman": "Her", "man": "His"}.get((gender or "").lower(), "Their")


def build_prompt(trigger: str | None, gender: str | None, motion: str) -> str:
    """`<trigger, gender>\\n<scene>\\n<motion> <still>` — the Prompt recipe's <TRIGGER>/<SCENE>/<PROMPT>."""
    who = {"woman": "woman", "man": "man"}.get((gender or "").lower(), "person")
    phrase = ", ".join(x for x in (trigger, gender) if x)
    scene = f"A close-up portrait of a {who} looking straight into the camera, plain background, soft even light."
    still = STILL.replace("Her ", f"{pronoun(gender)} ", 1)
    return f"{phrase}\n{scene}\n{motion.strip()} {still}".strip()
