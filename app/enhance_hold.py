"""Hold a segment's claim while the enhanced prompt its <PROMPT> needs is on its way (wanly-api#507).

WHAT WAS WRONG

    <PROMPT> is filled at the claim from the job's start image's saved enhanced prompt, and
    DROPPED when there is none (segments._resolve_enhanced_prompt). Click Enhance prompt, make a
    job straight away, and an idle worker claims it seconds later -- before the enhancer (~25 s
    warm, longer behind a bulk batch) has saved anything. The job rendered without the prompt
    it was made for, silently. A Re-enhance had the same race with the OLD text.

WHAT THIS DOES

    The claim does not hand out a PENDING segment whose prompt still carries a bare <PROMPT>
    while that segment's prompt image is

      * QUEUED   -- in the enhancer's line (image_meta.enhance_queued_at set), or
      * FAILED   -- its last enhance failed (enhance_error set) and it has no enhanced prompt
                    saved. Rendering without would be the silent drop this exists to stop.

    That is the whole mechanism: a WHERE clause on the claim. Unlike the caption hold
    (app/caption_hold.py) there is no new status, waiter or sweep, because the enhance queue is
    already durable on image_meta: the moment the enhancer writes the row, the next poll can
    claim the segment. A restart loses nothing.

    A failure is cleared from the console's enhance queue chip: Retry queues it again (back to
    QUEUED), Dismiss clears the error, and the segment then renders with <PROMPT> dropped --
    the person chose that, so it is no longer silent.

    Job detail says which state a segment is in (SegmentResponse.enhance_wait, blocked_reason),
    so a held segment never looks like an empty queue (console#422's rule).
"""
from __future__ import annotations

from sqlalchemy import and_, case, exists, func, not_, or_, select

from app.models import ImageMeta, Job, Segment

PROMPT_PLACEHOLDER = "<PROMPT>"

QUEUED = "queued"
FAILED = "failed"


def state(meta: ImageMeta | None) -> str | None:
    """QUEUED, FAILED or None for one image's row -- the Python twin of claim_gate()."""
    if meta is None:
        return None
    if meta.enhance_queued_at is not None:
        return QUEUED
    if meta.enhance_error and not (meta.enhanced_prompt or "").strip():
        return FAILED
    return None


def wants(prompt: str | None) -> bool:
    return PROMPT_PLACEHOLDER in (prompt or "")


def _prompt_image_sql():
    """segments._prompt_image as SQL: segment 0 its own start image (else the job's); a
    continuation always the job's."""
    return case(
        (Segment.index == 0, func.coalesce(Segment.start_image, Job.starting_image)),
        else_=Job.starting_image,
    )


def claim_gate() -> tuple:
    """WHERE terms for the claim query (which joins Job): skip segments held on an enhance."""
    held_image = exists(
        select(ImageMeta.path).where(
            ImageMeta.path == _prompt_image_sql(),
            or_(
                ImageMeta.enhance_queued_at.is_not(None),
                and_(
                    ImageMeta.enhance_error.is_not(None),
                    func.coalesce(func.btrim(ImageMeta.enhanced_prompt), "") == "",
                ),
            ),
        )
    )
    # Segment.prompt is NOT NULL, but coalesce keeps a NULL from turning NOT(...) into NULL.
    return (not_(and_(func.coalesce(Segment.prompt, "").contains(PROMPT_PLACEHOLDER),
                      held_image)),)


def reason(st: str, meta: ImageMeta, path: str, position: int | None, depth: int) -> str:
    """The blocked_reason text for a held segment."""
    name = path.rsplit("/", 1)[-1]
    if st == QUEUED:
        where = f" (#{position} of {depth} in the enhance queue)" if position else ""
        return f"Waiting for Enhance prompt on {name}{where}"
    return (f"Enhance prompt failed on {name}: {meta.enhance_error}. Retry it from the enhance "
            f"queue chip, or Dismiss there to render without it.")
