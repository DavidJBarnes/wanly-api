"""<PROMPT>: a recipe filled from the job's start image's enhanced prompt (wanly-api#484).

The enhanced prompt is the Sulphur-2 enhancer's scene script, made by clicking Enhance prompt in
the image modal (wanly-api#480). <PROMPT> is CACHE-ONLY and resolved at the claim, from the
JOB's start image for every segment -- a continuation's own frame is generated and never
enhanced. A miss drops the placeholder, exactly as an unresolved <SCENE> is dropped.
"""
from unittest.mock import AsyncMock, patch

import pytest

from app.config import settings
from app.models import ImageMeta, SegmentStatus
from app.routes.segments import PROMPT_PLACEHOLDER, SCENE_PLACEHOLDER, _resolve_enhanced_prompt
from app.routes.wildcards import RESERVED_WILDCARD_NAMES
from tests.test_claim_endpoint import _claim, _job, _segment

REPO = f"s3://{settings.s3_images_bucket}/2026-10-28/00001.png"
SCRIPT = "She sits on the sofa, then slowly leans forward toward the camera, smiling."


async def _enhanced(db, path=REPO, text=SCRIPT, scene=None):
    db.add(ImageMeta(path=path, enhanced_prompt=text, scene_description=scene))
    await db.flush()


class TestResolve:
    @pytest.mark.asyncio
    async def test_a_saved_prompt_is_swapped_in(self, db):
        await _enhanced(db)
        assert await _resolve_enhanced_prompt(db, f"k3llydw, {PROMPT_PLACEHOLDER}", REPO) == \
            f"k3llydw, {SCRIPT}"

    @pytest.mark.asyncio
    async def test_a_miss_drops_the_placeholder_and_its_separator(self, db):
        out = await _resolve_enhanced_prompt(db, f"k3llydw, {PROMPT_PLACEHOLDER}, slow", REPO)
        assert out == "k3llydw, slow"

    @pytest.mark.asyncio
    async def test_a_blank_saved_prompt_is_a_miss(self, db):
        await _enhanced(db, text="   ")
        assert await _resolve_enhanced_prompt(db, f"k3llydw, {PROMPT_PLACEHOLDER}", REPO) == \
            "k3llydw"

    @pytest.mark.asyncio
    async def test_no_image_is_a_miss(self, db):
        assert await _resolve_enhanced_prompt(db, f"a, {PROMPT_PLACEHOLDER}", None) == "a"

    @pytest.mark.asyncio
    async def test_it_never_calls_the_enhancer(self, db):
        """Cache-only: a claim polled with a 10 s timeout cannot wait ~25 s for a script."""
        with patch("app.prompt_enhance.enhance", new=AsyncMock()) as enhance:
            await _resolve_enhanced_prompt(db, PROMPT_PLACEHOLDER, REPO)
        enhance.assert_not_called()

    @pytest.mark.asyncio
    async def test_placeholder_tags_in_the_script_are_scrubbed(self, db):
        """Model output. <SCENE> is filled after <PROMPT>, so a literal tag would be taken
        for a placeholder."""
        await _enhanced(db, text="She waits <SCENE> by the <prompt>door</prompt>.")
        out = await _resolve_enhanced_prompt(db, PROMPT_PLACEHOLDER, REPO)
        assert out == "She waits  by the door."

    def test_prompt_is_a_reserved_wildcard_name(self):
        """Wildcards resolve at submit; a wildcard named PROMPT would substitute first."""
        assert "PROMPT" in RESERVED_WILDCARD_NAMES


@pytest.mark.asyncio
class TestClaim:
    async def test_segment_0_gets_the_start_images_prompt(self, db):
        await _enhanced(db)
        job = await _job(db, starting_image=REPO)
        seg = await _segment(db, job)
        seg.prompt = f"k3llydw, {PROMPT_PLACEHOLDER}"
        await db.flush()
        resp = await _claim(db)
        assert resp.status_code == 200, resp.text
        assert resp.json()["prompt"] == f"k3llydw, {SCRIPT}"
        await db.refresh(seg)
        assert seg.prompt == f"k3llydw, {SCRIPT}"      # the row records what ran

    async def test_a_continuation_uses_the_jobs_start_image_not_its_own_frame(self, db):
        await _enhanced(db)
        job = await _job(db, starting_image=REPO)
        await _segment(db, job, 0, status=SegmentStatus.COMPLETED,
                       last_frame=f"s3://{settings.s3_jobs_bucket}/j/seg0-last.png")
        seg = await _segment(db, job, 1)
        seg.prompt = f"k3llydw, {PROMPT_PLACEHOLDER}"
        await db.flush()
        resp = await _claim(db)
        assert resp.status_code == 200, resp.text
        assert resp.json()["prompt"] == f"k3llydw, {SCRIPT}"

    async def test_a_miss_with_words_of_its_own_renders_without_it(self, db):
        job = await _job(db, starting_image=REPO)
        seg = await _segment(db, job)
        seg.prompt = f"k3llydw, {PROMPT_PLACEHOLDER}, she leans in slowly"
        await db.flush()
        resp = await _claim(db)
        assert resp.status_code == 200, resp.text
        assert resp.json()["prompt"] == "k3llydw, she leans in slowly"

    async def test_a_miss_that_leaves_only_the_trigger_is_refused_and_restored(self, db):
        """"<TRIGGER>, <PROMPT>" with no enhanced prompt is an empty render. caption_failed,
        and the literal placeholder comes back so Retry after an Enhance fills it."""
        job = await _job(db, starting_image=REPO)
        seg = await _segment(db, job)
        seg.prompt = f"<TRIGGER>, {PROMPT_PLACEHOLDER}"
        await db.flush()
        resp = await _claim(db)
        assert resp.status_code == 200 and resp.json() is None, resp.text   # nothing handed out
        await db.refresh(seg)
        assert seg.status == SegmentStatus.CAPTION_FAILED
        assert seg.prompt == f"<TRIGGER>, {PROMPT_PLACEHOLDER}"
        assert PROMPT_PLACEHOLDER in seg.error_message

    async def test_prompt_and_scene_fill_together(self, db):
        await _enhanced(db, scene="a woman on a red sofa")
        job = await _job(db, starting_image=REPO)
        seg = await _segment(db, job)
        seg.prompt = f"{SCENE_PLACEHOLDER}. {PROMPT_PLACEHOLDER}"
        await db.flush()
        resp = await _claim(db)
        assert resp.status_code == 200, resp.text
        assert resp.json()["prompt"] == f"a woman on a red sofa. {SCRIPT}"
