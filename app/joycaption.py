"""JoyCaption — describing a start frame so a prompt can stop contradicting it.

WHY THIS EXISTS (console#405)
    An LTX prompt is two halves: what the scene IS, then what HAPPENS. Recipes are
    character- and start-frame-agnostic by design, so their static half is a generic guess
    about a frame they have never seen. A validated pose reads "a woman kneeling in front of
    a nude man" while the actual start image is a clothed woman sitting on a sofa.

    The vision model owns the static half. The recipe keeps the arc.

WHY AN UNCENSORED MODEL
    A stock captioner refuses this material or sanitises it into uselessness. Measured on a
    real continuation frame, JoyCaption returns "performs oral sex on a man standing by a
    pool ... she looks up at him" — the pose, the gaze and the act, which is exactly what a
    prompt needs and exactly what a general-purpose model will not say.

WHERE IT RUNS
    The 2070, not a render box. The 3090 sits at ~23 of 24 GB while rendering. The 2070 also
    hosts Automatic1111, which is why keep_alive is short: a resident 5.5 GB model would
    starve image generation.
"""
import base64
import hashlib
import logging

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


class CaptionError(RuntimeError):
    """The captioner could not be reached, or refused. Never fatal to a render."""


# The verbosity presets exposed in Settings.
#
# Every one of them ends with the same two suppressions, and they are not optional: the
# model will otherwise describe overlay text, picture frames and camera angles. Measured on
# a real frame it reported cursive text in the corner and confabulated it into a person's
# name. That is true of the image and garbage in a render prompt.
#
# GAZE AND EXPRESSION ARE REQUESTED EXPLICITLY in every preset but `raw`. A plain "describe
# this image" omits them, and "she is looking at the viewer" is load-bearing in these
# prompts — it is the difference between a subject engaging the camera and one staring past
# it. That single addition is what separated a usable caption from a nearly-usable one.
_SUPPRESS = (
    " State only what is visible. Do not mention the photograph, the camera, image quality, "
    "any text or writing in the image, or the picture frame."
)

CAPTION_STYLES: dict[str, str] = {
    # ~25 words. For recipes whose arc is already long, where the scene half should not
    # compete with it.
    "terse": (
        "In under 25 words and starting directly with the subject, describe who is in this "
        "image, what they are wearing, their pose, and where they are looking." + _SUPPRESS
    ),
    # ~40 words. The default, and the one that tested best.
    "standard": (
        "In under 40 words and starting directly with the subject, describe: who is in this "
        "image and what they look like, what they are wearing, their pose and where their "
        "hands are, their expression and where they are looking, and the setting."
        + _SUPPRESS
    ),
    # ~80 words. More of the scene, at the cost of weighing more heavily against the arc.
    "rich": (
        "In under 80 words and starting directly with the subject, describe in detail: who "
        "is in this image and what they look like, their hair, what they are wearing, their "
        "pose and the position of their hands and body, their expression and where they are "
        "looking, and the setting and lighting." + _SUPPRESS
    ),
    # JoyCaption's own voice, unshaped. Longest and most natural, but it WILL describe
    # overlay text and framing, because nothing here tells it not to.
    "raw": "Write a descriptive caption for this image.",
}
DEFAULT_STYLE = "standard"


def instruction_for(style: str, custom: str = "") -> str:
    """The instruction to send. A non-empty custom instruction always wins.

    Custom is the escape hatch: the presets encode what tested well, but the person tuning
    prompts knows their material better than a default does.
    """
    if custom and custom.strip():
        return custom.strip()
    return CAPTION_STYLES.get(style, CAPTION_STYLES[DEFAULT_STYLE])


def image_key(image_bytes: bytes, instruction: str) -> str:
    """Cache key: the image AND the instruction that will be applied to it.

    Keyed on CONTENT, not path — the same frame is referenced from several places, and a
    retry must reproduce the caption its first attempt used. Without that, retrying a failed
    segment re-captions, gets different words, and renders something different from what
    failed, which quietly breaks the meaning of "retry".

    The instruction is in the key because changing the style should produce a new caption
    rather than serve the old one.
    """
    h = hashlib.sha256()
    h.update(image_bytes)
    h.update(b"\x00")
    h.update(instruction.encode())
    return h.hexdigest()


async def describe(image_bytes: bytes, instruction: str) -> str:
    """Caption one image. Raises CaptionError; callers must treat that as non-fatal."""
    payload = {
        "model": settings.joycaption_model,
        "prompt": instruction,
        "images": [base64.b64encode(image_bytes).decode()],
        "stream": False,
        "keep_alive": settings.joycaption_keep_alive,
    }
    url = f"{settings.joycaption_url.rstrip('/')}/api/generate"
    try:
        async with httpx.AsyncClient(timeout=settings.joycaption_timeout_s) as client:
            resp = await client.post(url, json=payload)
    except httpx.HTTPError as e:
        raise CaptionError(f"captioner unreachable at {settings.joycaption_url}: {e}") from e
    if resp.status_code != 200:
        raise CaptionError(f"captioner returned {resp.status_code}: {resp.text[:200]}")

    text = (resp.json().get("response") or "").strip()
    if not text:
        raise CaptionError("captioner returned an empty caption")
    return _tidy(text)


def _tidy(text: str) -> str:
    """Make a caption safe to splice into the middle of a prompt.

    The caption lands between a trigger and an arc:

        <TRIGGER>, <SCENE>, she grips his penis with one hand...

    A trailing full stop would end the sentence mid-prompt, and a leading "The image shows"
    reads as instruction to render an image of an image. Newlines would break the single
    prompt line the encoder receives.
    """
    text = " ".join(text.split())
    for lead in ("This image shows ", "The image shows ", "This image depicts ",
                 "The image depicts ", "This is an image of ", "The photo shows "):
        if text.lower().startswith(lead.lower()):
            text = text[len(lead):]
            break
    return text.rstrip(" .")
