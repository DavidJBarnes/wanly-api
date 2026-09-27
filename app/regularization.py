"""Regularization pools: generic people rendered on the base model (wanly-api#352).

WHY. A character LoRA is trained on captions that start "<trigger>, woman". With nothing
else in the set, "woman" itself drifts toward the character -- which is why every woman in a
render came out looking like the character and every man like David. Regularization images
are the counterweight: the same class word over many DIFFERENT people, so the model is told
what "woman" still means while it learns what the trigger means.

WHY RENDERED FROM 10EROS. The pool has to look like what the base model already makes of
"woman" -- that is the prior being protected. Rendering it from the very checkpoint the
LoRA trains and renders against is the most direct way to get exactly that prior, and it
needs nothing but GPU time on a box that is otherwise idle.

WHY TEXT-TO-VIDEO, 25 FRAMES. There is no start frame to give it (that would be a person),
and a still is all that is kept: the segment's last frame. One second is the shortest the
stack renders, and the frame count is the cost.

WHY SO VARIED. A pool of one face at one framing regularizes toward that face. Framing, age,
ethnicity, setting, lighting and clothing are drawn independently per render so no two
share much but the class word. Age is stated here -- unlike in a training caption -- because
this is what the base model is ASKED to draw, and asking for a spread is the whole point.
"""
import random

from app.ltx_stack import LTX_STACK

#: Where a pool's renders sit in the queue: behind everything. The pool is several GPU-hours
#: of work nobody is waiting on, meant for a box that is otherwise idle, so a render somebody
#: IS waiting for must never queue behind it. Job creation (routes/jobs.py) takes its "bottom
#: of the queue" from the jobs BELOW this line, so a new job still lands ahead of a pool.
REG_PRIORITY_BASE = 1_000_000

#: One second at the stack's 24 fps, rounded to LTX's 8n+1 frame grid.
REG_FRAMES = 25
#: Mostly portrait, like the character sets they sit beside; some landscape so the pool does
#: not teach that the class word implies a vertical frame. Both are multiples of 64.
PORTRAIT = (832, 1216)
LANDSCAPE = (1216, 832)
LANDSCAPE_SHARE = 0.25

FRAMINGS = [
    "a close-up portrait", "a head-and-shoulders shot", "a medium shot",
    "a medium close-up", "a waist-up shot", "a full-body wide shot", "a three-quarter shot",
]
AGES = ["in her twenties", "in her thirties", "in her forties", "in her fifties",
        "in her sixties"]
ETHNICITIES = ["white", "Black", "East Asian", "South Asian", "Latina", "Middle Eastern",
               "Southeast Asian", "mixed-race"]
SETTINGS = [
    "a sunlit kitchen", "a city street", "a park in autumn", "a quiet library",
    "a busy cafe", "an office with large windows", "a beach at dusk", "a living room",
    "a subway platform", "a garden", "a hotel lobby", "a mountain trail", "a bedroom",
    "a rooftop terrace at night", "a supermarket aisle", "a studio with a plain backdrop",
]
LIGHTING = [
    "soft window light", "golden hour sunlight", "overcast daylight", "warm lamplight",
    "cool fluorescent light", "neon light at night", "harsh midday sun",
    "soft studio lighting",
]
CLOTHING_WOMAN = [
    "a denim jacket", "a summer dress", "a grey hoodie", "a business suit", "a knit sweater",
    "a white t-shirt and jeans", "a leather jacket", "a raincoat", "a blouse and skirt",
    "workout clothes",
]
CLOTHING_MAN = [
    "a denim jacket", "a grey hoodie", "a business suit", "a knit sweater",
    "a white t-shirt and jeans", "a leather jacket", "a raincoat", "a flannel shirt",
    "a polo shirt", "workout clothes",
]
ACTIONS = [
    "looking at the camera", "glancing to the side", "smiling softly", "laughing",
    "with a neutral expression", "looking down", "turning their head slowly",
    "talking to someone off camera",
]


def reg_prompt(reg_class: str, rng: random.Random) -> str:
    """One varied prompt for the class word. Pure given the generator, for tests."""
    age = rng.choice(AGES)
    if reg_class == "man":
        age = age.replace("her", "his")
        clothing = rng.choice(CLOTHING_MAN)
    else:
        clothing = rng.choice(CLOTHING_WOMAN)
    return (f"{rng.choice(FRAMINGS)} of a {rng.choice(ETHNICITIES)} {reg_class} {age} "
            f"wearing {clothing}, {rng.choice(ACTIONS)}, in {rng.choice(SETTINGS)}, "
            f"{rng.choice(LIGHTING)}, realistic, natural skin texture, subtle natural movement")


def reg_size(rng: random.Random) -> tuple[int, int]:
    return LANDSCAPE if rng.random() < LANDSCAPE_SHARE else PORTRAIT


def reg_recipe(reg_class: str) -> dict:
    """The segment's ltx_recipe: a named recipe (the engine's recipe path) with NO person.

    `characters` is empty and `char_lora` is "none", the value every reader already treats
    as "no LoRA" -- the claim's model requirements, the daemon's LoRA list, the engine's
    loader. The checkpoint is named explicitly rather than left to the stack default: this
    pool must come from the same base the LoRA trains against, and a later change to the
    default must not quietly regenerate part of a pool from something else.

    Keyframes are left to the daemon, which sends [] for a segment with no start image; the
    engine's recipe path renders that as text-to-video at the request's width/height
    (wanly-gpu-docker, alongside this change).
    """
    return {
        "recipe": f"Regularization ({reg_class})",
        "characters": [],
        "character": None,
        "trigger": None,
        "char_lora": "none",
        "frames": REG_FRAMES,
        "img_compression": LTX_STACK["img_compression"],
        "content_loras": [],
        "checkpoint": LTX_STACK["checkpoint"],
        "edited": [],
    }


def reg_tag(dataset_id) -> str:
    """The job tag that ties a render to the pool it is for.

    A tag rather than a column: jobs already carry tags, the jobs list already filters on
    them whole-tag (tag_filter.tag_clause), and a pool is a few hundred throwaway jobs -- a
    foreign key for them would be schema for bookkeeping.
    """
    return f"reg-{dataset_id}"
