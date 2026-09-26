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
    Since wanly-gpu-docker#83, inside the GPU container on the 3090 as the image-description
    service -- the same card the render stack uses, which sits at ~23 of 24 GB while
    rendering. keep_alive is short so the model holds VRAM only during a caption, and an
    INTERACTIVE caption is refused while that box is rendering (busy_render_beside_the_captioner)
    until the local VRAM lease lands. Before #83 it ran on the 2070 beside Automatic1111;
    _yield_the_gpu is that arrangement's half of the sharing and still applies wherever
    a1111_url points at a box that also captions.

    This module keeps its file name; the service it talks to is image-description.
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


# ---------------------------------------------------------------------------------------
# Motion descriptions (wanly-api#326)
#
# The other half of the caption: the image read as the FIRST FRAME of a 10-second clip,
# which is the prompt an image-to-video model actually wants. Prototyped on the 2070 with
# qwen2.5vl on 2026-09-15/16; the findings below shaped these prompts and are load-bearing.
#
# 1. WITHOUT AN ANTI-HEDGE THE MODEL DESCRIBES A STILL PHOTO. Untuned it returned "her
#    expression remains neutral... the camera remains steady" for every frame. "The action
#    implied by the frame begins and continues throughout" plus an explicit ban on
#    remains-still hedging is what produced real motion language.
# 2. DIRECTION AND AMPLITUDE MUST BE ASKED FOR BY NAME ("up and down, deeper or shallower,
#    in or out"). "Describe the motion" gets adjectives; the named axes get the verbs a
#    video prompt needs.
# 3. BEATS DON'T WORK. A 0-3s/3-7s/7-10s format reads beautifully but LTX has no timestamp
#    syntax; one flowing paragraph is the form it can consume.
# 4. STYLE TALK CROWDS OUT ACTION. The variant that asked the model to also choose a
#    capture style answered with the man "remains still". The style sentence therefore sits
#    at the END, after the action has claimed the word budget.
# 5. GROUNDING ON THE STATIC CAPTION stops the two paragraphs contradicting each other
#    (ungrounded, it invented hip motion on a hand-motion frame; grounded on a static half
#    that says "holding his penis with both hands", the motion kept the hands).
# ---------------------------------------------------------------------------------------

MOTION_BASE = (
    "Write a text-to-video prompt for a 10-second clip that begins with this exact frame. "
    "The main action implied by the frame begins and continues throughout the clip. "
    "Describe the action in explicit physical direction and amplitude: who moves, in which "
    "direction (toward or away from the other person, deeper or shallower, up and down, "
    "hips rocking in or out), how far and how fast, and how the rhythm changes over the ten "
    "seconds. Cover how each person's body follows the motion - arching back, head tilting, "
    "hands repositioning, gripping - and how expressions and eye contact evolve. Add "
    "secondary motion: hair sway, skin, fabric stretching. "
)

MOTION_TAIL = (
    " Single continuous shot, no cuts, no scene change. Do not hedge with 'remains still' "
    "or 'slight shift' unless a person genuinely is static. One flowing paragraph, under "
    "110 words, starting with the main subject."
)

#: Capture-style sentence, spliced in before the tail. Measured: a style preset changes the
#: flavour of the output without stealing word budget from the action when it sits at the
#: end ("none" exists for exactly that reason).
MOTION_STYLE_PRESETS: dict[str, str] = {
    # The house style for this material: consumer-cam realism.
    "handheld": ("Render it as handheld footage with natural micro-shake and small "
                 "reflexive reframing. "),
    "amateur": ("Render it as amateur consumer-camera footage: micro-shake, available "
                "indoor light, no cinematic polish. "),
    "cinematic": ("Render it as cinematic footage: slow controlled push-in, shallow depth "
                  "of field, filmic color. "),
    "static": "Camera locked off on a tripod, no movement at all. ",
    # No sentence at all — the model's own default, and the option that leaves the most
    # words for the action.
    "none": "",
}
MOTION_DEFAULT_STYLE = "handheld"

#: Identity lock + the LTX-2.3 audio half. The soundscape sentence is free signal: LTX 2.3
#: renders synced sound, and "soft gasps and moans, room tone" is promptable.
MOTION_GROUNDING = "Keep both people's faces, bodies and wardrobe exactly as in the frame for the whole clip. End with one sentence on the soundscape over the ten seconds."


def motion_instruction_for(style: str, custom: str = "", scene: str = "") -> str:
    """The motion instruction, grounded on the static caption when one is given.

    Same custom-wins rule as instruction_for: a non-empty custom instruction is the whole
    prompt, grounding and all. Appending the identity lock to it would not be an escape
    hatch, it would be a second opinion about a decision the caller already made.

    `scene` is the static caption from the first call of the same session; it anchors
    identity and wardrobe so the motion paragraph cannot contradict what the person just
    read and accepted.
    """
    if custom and custom.strip():
        return custom.strip()
    style_sentence = MOTION_STYLE_PRESETS.get(style,
                                              MOTION_STYLE_PRESETS[MOTION_DEFAULT_STYLE])
    prompt = MOTION_BASE + style_sentence + MOTION_TAIL
    if scene and scene.strip():
        prompt = (f"Scene: {scene.strip()}\n\n{prompt}\n\n{MOTION_GROUNDING} "
                  "Do not restate the scene description.")
    return prompt


async def describe_motion(image_bytes: bytes, scene_description: str, style: str,
                          custom: str = "", base_url: str | None = None) -> tuple[str, str]:
    """The motion paragraph for an image, grounded on its static caption.

    Returns (motion, instruction_used) — same provenance rule as the static caption: the
    row must say how each half was made. Raises CaptionError; callers treat that as
    non-fatal to the static half.
    """
    instruction = motion_instruction_for(style, custom, scene_description)
    return await describe(image_bytes, instruction, base_url), instruction


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


async def _yield_the_gpu() -> bool:
    """Ask Automatic1111 for the card back. True if it actually let go.

    The 2070 is shared, and the sharing was one-directional: joycaption_keep_alive is 5s so
    a caption releases VRAM the moment it is done, while A1111 holds its checkpoint until
    told otherwise — idle or not. That is enough on its own to abort a caption.

    MEASURED on the 2070, an 8192 MiB card:

        JoyCaption at load    ~5970 MiB   3992 weights + 512 KV + 669 compute + 800 vision
        A1111 idle, loaded    ~1720-1850 MiB

    They do not fit. The load gets as far as the vision projector, the last 800 MiB
    cudaMalloc fails, ggml aborts the runner, and ollama reports it as

        llama runner process has terminated: %!w(<nil>)

    which names neither the GPU nor the memory, and reads like a broken model. Both times it
    happened here the card was NOT busy — A1111 was idle with a checkpoint resident, which
    is simply its state between generations.

    A1111 reloads its checkpoint from RAM on its next generation, a few seconds. That is the
    right trade against a caption that cannot run at all — but only when it buys something,
    which is why this is called on failure rather than before every caption. A burst of
    captions coalesces inside the 5s keep_alive and never disturbs A1111 at all.

    NOTHING HERE IS FATAL. An absent A1111 is the normal case anywhere else and returns
    False, exactly like one that refuses: both mean the retry has no reason to behave
    differently, so there is no retry.
    """
    base = settings.a1111_url.rstrip("/")
    if not base:
        return False
    try:
        async with httpx.AsyncClient(timeout=settings.a1111_yield_timeout_s) as client:
            # Never interrupt work in progress. Unloading mid-generation would take down
            # somebody's image to caption a frame, and the caption is the less urgent of the
            # two — it has a fallback, the generation does not.
            busy = await client.get(f"{base}/sdapi/v1/progress")
            if busy.status_code == 200 and (busy.json().get("state") or {}).get("job_count"):
                logger.info("A1111 is generating; leaving its checkpoint alone")
                return False
            resp = await client.post(f"{base}/sdapi/v1/unload-checkpoint")
            if resp.status_code != 200:
                logger.warning("A1111 refused to unload: %s", resp.status_code)
                return False
    except httpx.HTTPError as e:
        logger.info("A1111 not reachable at %s (%s) — nothing to free", base, e)
        return False
    logger.info("A1111 released its checkpoint; retrying the caption")
    return True


class CaptionerBusy(CaptionError):
    """The box that captions is rendering right now; ask again when it finishes."""


def captioner_host() -> str:
    """The host part of image_description_url: `3090.zero` for http://3090.zero:11434."""
    from urllib.parse import urlsplit
    return (urlsplit(settings.image_description_url).hostname or "").lower()


def render_worker_beside_the_captioner(workers) -> "object | None":
    """The render-capable worker row that shares a card with the captioner, or None.

    Matched by HOST: the row whose friendly_name is the host in image_description_url.
    Not by what a row says it provides -- the 3090 provides image-description, so a
    provides-match picked the 3090's row even when the URL pointed at the 2070, and a
    caption during a render was then sent to the box that was rendering. A row that cannot
    render is never a collision: a captioner-only box has nothing to wait for.
    """
    from app.enums import WorkerKind, worker_can
    host = captioner_host()
    for w in workers:
        if not worker_can(w, WorkerKind.RENDER):
            continue
        if (w.friendly_name or "").lower() == host:
            return w
    return None


#: The box's MODE, cached for a moment. A caption asks once per request and a batch is
#: dozens of requests; the mode changes on a human action, so a few seconds of staleness
#: costs nothing and a call per caption would put the control API in the hot path.
_MODE_CACHE: dict[str, tuple[float, str | None]] = {}
_MODE_TTL_S = 5.0


async def _render_mode(worker) -> str | None:
    """What mode the box beside the captioner is in, or None if it will not say.

    Unreadable is NOT treated as render mode: an older container has no /mode at all, and
    refusing every caption on a box that simply cannot answer would be worse than the
    contention this prevents.
    """
    import time

    import httpx

    from app.config import settings

    name = worker.friendly_name
    hit = _MODE_CACHE.get(name)
    now = time.monotonic()
    if hit and now - hit[0] < _MODE_TTL_S:
        return hit[1]
    mode = None
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(
                f"http://{name}:{settings.worker_control_port}/health")
        # A degraded box answers 503 and its body is still the truth -- a container with a
        # service deliberately stopped is exactly that shape.
        mode = (r.json() or {}).get("mode")
    except Exception:
        mode = None
    _MODE_CACHE[name] = (now, mode)
    return mode


async def busy_render_beside_the_captioner(db) -> str | None:
    """The friendly_name of the render worker sharing the captioner's card, when captioning
    on it would collide with rendering; else None. The interactive caption routes refuse on it.

    TWO REASONS TO REFUSE, and the second is the one that makes a MODE mean anything.

        online-busy   a segment is being rendered right now.

        render mode   the box is SET to render. One GPU does one job at a time -- that is
                      the entire point of the mode, not a side effect of which processes
                      happen to be up. Keying only on online-busy left render mode with no
                      teeth: between claims the box looked idle, captions were accepted, and
                      they raced the render stack for the card. Caption mode has always
                      blocked renders (there is no daemon to claim), so this is the missing
                      half of the same rule, not a new one.

    Refuse, not wait: a 720p render is ~30 minutes and the request sits behind a 60 s proxy
    timeout, so a wait would surface as a 504 that reads like a dead captioner. A clear
    "3090.zero is in render mode" the person can act on is worth more. Claim-time <SCENE>
    resolution does NOT go through this: the worker has just been handed the segment and
    has not loaded the render yet, and a failed caption there is non-fatal by design.
    """
    from sqlalchemy import select
    from app.models import Worker
    rows = (await db.execute(select(Worker).where(Worker.status != "offline"))).scalars().all()
    w = render_worker_beside_the_captioner(rows)
    if w is None:
        return None
    if w.status == "online-busy":
        return w.friendly_name
    if await _render_mode(w) == "ltx-engine":
        return w.friendly_name
    return None


def captioner_for(busy: str | None, interactive: bool) -> str | None:
    """Which captioner URL to use, or None to refuse.

    The 3090's captioner shares its card with the render stack (wanly-gpu-docker#83), so:
      * interactive, box idle      -> the primary
      * interactive, box rendering -> the fallback if there is one, else refuse (None)
      * claim-time                 -> the fallback first if there is one: the claiming box
                                      is about to load a 23 GB render, and a caption that
                                      races it timed out on the first night; else the primary
    """
    primary = settings.image_description_url
    fallback = (settings.image_description_fallback_url or "").strip()
    if not interactive:
        return fallback or primary
    if busy:
        return fallback or None
    return primary


async def describe(image_bytes: bytes, instruction: str, base_url: str | None = None) -> str:
    """Caption one image. Raises CaptionError; callers must treat that as non-fatal."""
    base = (base_url or settings.image_description_url).rstrip("/")
    payload = {
        "model": settings.image_description_model,
        "prompt": instruction,
        "images": [base64.b64encode(image_bytes).decode()],
        "stream": False,
        "keep_alive": settings.image_description_keep_alive,
        # num_ctx is not a tuning knob here, it is the difference between the model being
        # resident and not -- see image_description_num_ctx. Sending nothing let ollama size
        # the context from VRAM and push a third of the layers onto the CPU.
        "options": {"num_ctx": settings.image_description_num_ctx},
    }
    url = f"{base}/api/generate"
    try:
        async with httpx.AsyncClient(timeout=settings.image_description_timeout_s) as client:
            resp = await client.post(url, json=payload)
            # ollama answers 500 for a runner that died loading, which on this box is
            # almost always the GPU rather than the model. Ask the other tenant to let go
            # and try once more; if nothing was freed, the second attempt would fail
            # identically, so it is not made.
            if resp.status_code == 500 and await _yield_the_gpu():
                resp = await client.post(url, json=payload)
    except httpx.HTTPError as e:
        raise CaptionError(f"captioner unreachable at {base}: {e!r}") from e
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
