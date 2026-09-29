"""Face edits -- expression, gaze, small head turns -- through the face-edit service (console#547).

The service (wanly-gpu-docker, SERVICES=face-edit, on the 2070 beside Automatic1111) runs
LivePortrait's ExpressionEditor: it WARPS the existing pixels through implicit keypoints, so the
person stays the same person, and every pixel outside the face mask is the source's own. That is
why it is the engine for identity datasets and why it comes first: Qwen-Image-Edit (phase 2,
#548) regenerates the frame and measurably de-ages the subject and halves skin texture on every
pass.

THE PRESETS LIVE HERE, once. The console reads them (GET /images/edit/presets) to draw its
buttons and sliders, and the service only ever receives numbers -- so a preset's meaning cannot
drift between the button that shows it and the model that applies it. The values are
keyframe-server's lexicon, which is what was tuned by eye on real faces.

Like face-crop, this is called inline: an edit is ~1 s on a free GPU and several seconds on the
CPU fallback, far below anything worth a queue row.
"""
from __future__ import annotations

import base64
import logging

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


#: The parameters, in the order the editor shows them. Ranges are the node's own
#: (ComfyUI-AdvancedLivePortrait ExpressionEditor.INPUT_TYPES); the service enforces the same
#: bounds, and the schema here refuses an out-of-range value before anything is fetched.
#:
#: `group` is presentation: "main" is the seven the editor always shows (#547's list: yaw,
#: pitch, roll, eye openness, smile, mouth open, eyebrow), "gaze" the two the look-* presets
#: drive, "more" the rest.
AXES: list[dict] = [
    {"key": "rotate_yaw", "label": "Turn (yaw)", "min": -20, "max": 20, "step": 0.5, "group": "main"},
    {"key": "rotate_pitch", "label": "Nod (pitch)", "min": -20, "max": 20, "step": 0.5, "group": "main"},
    {"key": "rotate_roll", "label": "Tilt (roll)", "min": -20, "max": 20, "step": 0.5, "group": "main"},
    {"key": "blink", "label": "Eye openness", "min": -20, "max": 5, "step": 0.5, "group": "main"},
    {"key": "smile", "label": "Smile", "min": -0.3, "max": 1.3, "step": 0.01, "group": "main"},
    {"key": "aaa", "label": "Mouth open", "min": -30, "max": 120, "step": 1, "group": "main"},
    {"key": "eyebrow", "label": "Eyebrow", "min": -10, "max": 15, "step": 0.5, "group": "main"},
    {"key": "pupil_x", "label": "Gaze left/right", "min": -15, "max": 15, "step": 0.5, "group": "gaze"},
    {"key": "pupil_y", "label": "Gaze up/down", "min": -15, "max": 15, "step": 0.5, "group": "gaze"},
    {"key": "wink", "label": "Wink", "min": 0, "max": 25, "step": 0.5, "group": "more"},
    {"key": "eee", "label": "Wide mouth (eee)", "min": -20, "max": 15, "step": 0.2, "group": "more"},
    {"key": "woo", "label": "Pursed mouth (woo)", "min": -20, "max": 15, "step": 0.2, "group": "more"},
]
AXIS_KEYS = tuple(a["key"] for a in AXES)

#: name -> (label, expression). From keyframe-server's _LEXICON, where each value was set by eye
#: on real faces: "smile", "eyes closed", "look *", "turn head *" are its entries verbatim, and
#: "big laugh" is what it resolved "big laugh" to (laugh x1.5, clamped). It had no entry for
#: the other three, so they are compositions of its terms: surprised = raised brows + wide eyes +
#: open mouth, serious = a slight frown, speaking = a part-open, slightly wide mouth (the one
#: preset with no measured precedent). Checked on a 1248x1824 test frame with the real model;
#: sliders are the escape hatch for any that reads too strong on a given face.
#:
#: "left"/"right" are the node's own sign convention (yaw is negated on entry, nodes.py:872),
#: carried over unchanged. Whether that is the SUBJECT's left or the viewer's is worth
#: confirming on the first real edit.
PRESETS: dict[str, tuple[str, dict[str, float]]] = {
    "smile": ("Smile", {"smile": 0.5}),
    "big_laugh": ("Big laugh", {"smile": 1.3, "aaa": 52.5}),
    "eyes_closed": ("Eyes closed", {"blink": -18}),
    "surprised": ("Surprised", {"eyebrow": 8, "blink": 4, "aaa": 45}),
    "serious": ("Serious", {"eyebrow": -3, "smile": -0.1}),
    "speaking": ("Speaking", {"aaa": 25, "eee": 4}),
    "look_left": ("Look left", {"pupil_x": -8}),
    "look_right": ("Look right", {"pupil_x": 8}),
    "look_up": ("Look up", {"pupil_y": 8}),
    "look_down": ("Look down", {"pupil_y": -8}),
    "turn_head_left": ("Turn head left", {"rotate_yaw": -12}),
    "turn_head_right": ("Turn head right", {"rotate_yaw": 12}),
}

#: What a preview comes back as. The service sits behind a home uplink (~0.6 MB/s measured on
#: the 3090's line): a full-size PNG is ~3 MB, ~5 s per slider drag; this is ~120 KB.
PREVIEW_MAX_EDGE = 1024


class FaceEditError(Exception):
    """Carries the HTTP status the route should answer with, and a sentence that says why."""

    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def resolve(preset: str | None, expression: dict[str, float | None] | None) -> dict[str, float]:
    """The numbers to send: the preset's, with any explicit value laid over it.

    That is how the editor works: clicking a preset moves the sliders to its values, and a
    drag afterwards changes one axis without losing the others. Only non-zero axes are kept, so
    the record of what was done reads as the edit, not twelve zeros.
    """
    params: dict[str, float] = {}
    if preset is not None:
        if preset not in PRESETS:
            raise FaceEditError(
                422, f"unknown preset {preset!r}; known: {', '.join(PRESETS)}")
        params.update(PRESETS[preset][1])
    for k, v in (expression or {}).items():
        if v is not None:
            params[k] = float(v)
    params = {k: v for k, v in params.items() if v}
    if not params:
        # Refused rather than sent: the service would answer the same, after an S3 fetch and a
        # round trip -- and "saved" an unchanged copy of the original is the worst outcome.
        raise FaceEditError(422, "nothing to apply: pick a preset or move a slider")
    return params


def _service_url() -> str:
    url = (settings.face_edit_url or "").strip().rstrip("/")
    if not url:
        raise FaceEditError(
            503, "no face-edit service is configured (face_edit_url is empty)")
    return url


async def edit(image_bytes: bytes, params: dict[str, float], *, preview: bool = False) -> dict:
    """Run one edit. Returns the service's JSON with `image` decoded to bytes.

    Raises FaceEditError with the status the caller should pass on: 503 for down or busy (the
    two things that fix themselves), 422 for this image or these numbers, 502 for anything the
    service should not have said.
    """
    base = _service_url()
    body: dict = {"image": base64.b64encode(image_bytes).decode(), "expression": params}
    if preview:
        body.update(format="jpeg", max_edge=PREVIEW_MAX_EDGE)
    try:
        async with httpx.AsyncClient(timeout=settings.face_edit_timeout_s) as client:
            r = await client.post(f"{base}/edit", json=body)
    except httpx.TimeoutException as e:
        raise FaceEditError(
            503, f"face-edit did not answer within {settings.face_edit_timeout_s}s at {base} "
                 f"— it may be busy or down ({type(e).__name__})") from e
    except httpx.HTTPError as e:
        raise FaceEditError(503, f"face-edit unreachable at {base}: {e!r}") from e

    if r.status_code != 200:
        try:
            detail = r.json().get("detail")
        except Exception:  # noqa: BLE001 -- a non-JSON error body is still an error
            detail = None
        detail = detail if isinstance(detail, str) else (r.text[:300] or r.reason_phrase)
        if r.status_code == 503:
            raise FaceEditError(503, f"face-edit is busy or not ready: {detail}")
        if r.status_code in (400, 413, 422):
            raise FaceEditError(422, f"face-edit refused this image: {detail}")
        raise FaceEditError(502, f"face-edit returned {r.status_code}: {detail}")

    out = r.json()
    try:
        out["image"] = base64.b64decode(out["image"])
    except Exception as e:
        raise FaceEditError(502, f"face-edit returned an unreadable image: {e}") from e
    logger.info("face edit on %s (%s) in %s ms: %s", out.get("device"),
                out.get("device_reason"), (out.get("timings_ms") or {}).get("total"), params)
    return out
