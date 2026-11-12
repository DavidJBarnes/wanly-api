"""The claim waits for a <PROMPT> image's enhance (wanly-api#507).

Before this, Enhance prompt clicked moments before a job was made raced the claim: an idle
worker took the segment, found no enhanced prompt saved, dropped <PROMPT> and rendered without
the script the job was made for.
"""
from datetime import datetime, timezone

import pytest

from app import enhance_hold
from app.config import settings
from app.models import ImageMeta, SegmentStatus
from app.routes.segments import PROMPT_PLACEHOLDER
from tests.test_claim_endpoint import _claim, _job, _segment, _user
from tests.test_claim_model_gate import _job_detail

REPO = f"s3://{settings.s3_images_bucket}/2026-11-12/00001.png"
OTHER = f"s3://{settings.s3_images_bucket}/2026-11-12/00002.png"
SCRIPT = "She leans toward the camera, then laughs and looks away."
NOW = datetime.now(timezone.utc)


async def _meta(db, path=REPO, *, queued=False, error=None, text=None):
    db.add(ImageMeta(path=path, enhance_queued_at=NOW if queued else None,
                     enhance_error=error, enhanced_prompt=text))
    await db.flush()


async def _held_job(db, start=REPO, prompt=f"k3llydw, {PROMPT_PLACEHOLDER}, slow", **kw):
    job = await _job(db, **kw, starting_image=start)
    seg = await _segment(db, job)
    seg.prompt = prompt
    await db.flush()
    return job, seg


class TestState:
    def test_queued(self):
        assert enhance_hold.state(ImageMeta(path=REPO, enhance_queued_at=NOW,
                                            enhanced_prompt=SCRIPT)) == enhance_hold.QUEUED

    def test_failed_with_nothing_saved(self):
        assert enhance_hold.state(ImageMeta(path=REPO, enhance_error="boom")) == \
            enhance_hold.FAILED

    def test_a_failed_reenhance_keeps_the_old_prompt_and_is_not_held(self):
        assert enhance_hold.state(ImageMeta(path=REPO, enhance_error="boom",
                                            enhanced_prompt=SCRIPT)) is None

    def test_no_row(self):
        assert enhance_hold.state(None) is None


@pytest.mark.asyncio
class TestClaim:
    async def test_a_queued_image_holds_the_segment(self, db):
        await _meta(db, queued=True)
        await _held_job(db)
        resp = await _claim(db)
        assert resp.status_code == 200 and resp.json() is None, resp.text

    async def test_a_reenhance_holds_even_with_an_old_prompt_saved(self, db):
        """The race with the OLD text: the new one is what the person asked for."""
        await _meta(db, queued=True, text="the old script")
        await _held_job(db)
        assert (await _claim(db)).json() is None

    async def test_it_is_claimed_once_the_enhancer_saves(self, db):
        await _meta(db, queued=True)
        _, seg = await _held_job(db)
        assert (await _claim(db)).json() is None
        meta = await db.get(ImageMeta, REPO)
        meta.enhance_queued_at = None
        meta.enhanced_prompt = SCRIPT
        await db.flush()
        body = (await _claim(db)).json()
        assert body["id"] == str(seg.id)
        assert body["prompt"] == f"k3llydw, {SCRIPT}, slow"

    async def test_a_failure_with_nothing_saved_holds(self, db):
        await _meta(db, error="no Action tag with a description")
        await _held_job(db)
        assert (await _claim(db)).json() is None

    async def test_dismissing_the_failure_renders_without(self, db):
        await _meta(db, error="boom")
        _, seg = await _held_job(db)
        meta = await db.get(ImageMeta, REPO)
        meta.enhance_error = None
        await db.flush()
        body = (await _claim(db)).json()
        assert body["id"] == str(seg.id)
        assert body["prompt"] == "k3llydw, slow"

    async def test_a_prompt_without_the_placeholder_is_not_held(self, db):
        await _meta(db, queued=True)
        _, seg = await _held_job(db, prompt="k3llydw, she leans in")
        assert (await _claim(db)).json()["id"] == str(seg.id)

    async def test_an_image_with_no_enhance_activity_is_not_held(self, db):
        """Today's behaviour stands: no enhance asked for, the placeholder is dropped."""
        _, seg = await _held_job(db)
        body = (await _claim(db)).json()
        assert body["id"] == str(seg.id) and body["prompt"] == "k3llydw, slow"

    async def test_the_line_moves_past_a_held_segment(self, db):
        await _meta(db, queued=True)
        user = await _user(db)
        await _held_job(db, user=user)
        _, free = await _held_job(db, start=OTHER, user=user)
        assert (await _claim(db)).json()["id"] == str(free.id)

    async def test_a_continuation_waits_on_the_jobs_start_image(self, db):
        await _meta(db, queued=True)
        job = await _job(db, starting_image=REPO)
        await _segment(db, job, 0, status=SegmentStatus.COMPLETED,
                       last_frame=f"s3://{settings.s3_jobs_bucket}/j/seg0-last.png")
        seg = await _segment(db, job, 1)
        seg.prompt = f"k3llydw, {PROMPT_PLACEHOLDER}"
        await db.flush()
        assert (await _claim(db)).json() is None

    async def test_segment_0_waits_on_its_own_start_image(self, db):
        await _meta(db, OTHER, queued=True)
        job = await _job(db, starting_image=REPO)
        seg = await _segment(db, job, start_image=OTHER)
        seg.prompt = f"k3llydw, {PROMPT_PLACEHOLDER}, slow"
        await db.flush()
        assert (await _claim(db)).json() is None


@pytest.mark.asyncio
class TestJobDetail:
    async def test_a_queued_hold_says_so_with_its_place(self, db):
        owner = await _user(db)
        await _meta(db, queued=True)
        job, _ = await _held_job(db, user=owner)
        seg = (await _job_detail(db, owner, job)).json()["segments"][0]
        assert seg["enhance_wait"] == "queued"
        assert seg["blocked_reason"] == \
            "Waiting for Enhance prompt on 00001.png (#1 of 1 in the enhance queue)"

    async def test_a_failed_hold_says_how_to_clear_it(self, db):
        owner = await _user(db)
        await _meta(db, error="the enhancer is down")
        job, _ = await _held_job(db, user=owner)
        seg = (await _job_detail(db, owner, job)).json()["segments"][0]
        assert seg["enhance_wait"] == "failed"
        assert "the enhancer is down" in seg["blocked_reason"]
        assert "Dismiss" in seg["blocked_reason"]

    async def test_nothing_to_say_when_not_held(self, db):
        owner = await _user(db)
        await _meta(db, text=SCRIPT)
        job, _ = await _held_job(db, user=owner)
        seg = (await _job_detail(db, owner, job)).json()["segments"][0]
        assert seg["enhance_wait"] is None and seg["blocked_reason"] is None
