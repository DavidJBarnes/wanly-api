"""The claim path reads the same saved description the console does (console#427).

Two things meet here. A repo image that was already described must not be described again —
that is 1.2-4.5s of 2070 time for words that already exist, and the model is
nondeterministic, so a second call would render with a DIFFERENT description from the one
the person read. And the console's `<scene>…</scene>` markers, which make a filled scene
rewritable while composing, must never reach the text encoder.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from app.config import settings
from app.models import ImageMeta
from app.routes.segments import (
    SCENE_PLACEHOLDER,
    _is_repo_image,
    _resolve_scene,
    _unwrap_scene,
)

REPO = f"s3://{settings.s3_images_bucket}/2026-09-05/00001.png"
GENERATED = "s3://wanly-jobs/some-job/seg0-last.png"
TEMPLATE = f"k3llydw, {SCENE_PLACEHOLDER}, she grips the edge of the sofa"


class TestUnwrapMarkers:
    """The console strips these before sending. This is the guarantee, not the promise —
    the daemon, a retried older prompt and anything that is not the console post here too."""

    def test_it_keeps_the_words_and_drops_the_markers(self):
        assert _unwrap_scene("k3llydw, <scene>a woman on a sofa</scene>, she grips") == (
            "k3llydw, a woman on a sofa, she grips"
        )

    def test_it_leaves_an_unfilled_placeholder_alone(self):
        # That one is _resolve_scene's job: it gets described, or dropped.
        assert _unwrap_scene(TEMPLATE) == TEMPLATE

    def test_it_is_case_insensitive_and_spans_newlines(self):
        assert _unwrap_scene("a <SCENE>two\nlines</SCENE> b") == "a two\nlines b"

    def test_two_regions_do_not_merge_into_one(self):
        # A greedy match would swallow everything between the first open and the last close,
        # taking the words in between with it.
        assert _unwrap_scene("<scene>a</scene> and <scene>b</scene>") == "a and b"

    def test_a_prompt_with_no_markers_is_untouched(self):
        assert _unwrap_scene("k3llydw, a woman on a sofa") == "k3llydw, a woman on a sofa"


class TestWhichImagesGetRows:
    """Only repo images. /images/search selects FROM image_meta and HEADs whatever path it
    finds, so a row for a generated frame would put that frame in the Image Repo."""

    def test_a_repo_image_does(self):
        assert _is_repo_image(REPO)

    def test_a_generated_continuation_frame_does_not(self):
        assert not _is_repo_image(GENERATED)

    def test_nothing_does(self):
        assert not _is_repo_image(None)
        assert not _is_repo_image("")


class TestResolveUsesTheCache:
    @pytest.mark.asyncio
    async def test_a_described_image_is_not_described_again(self, db):
        db.add(ImageMeta(path=REPO, scene_description="a woman in a red dress on a sofa",
                         scene_described_at=datetime.now(timezone.utc)))
        await db.flush()

        with patch("app.routes.segments.caption_image_bytes",
                   new=AsyncMock()) as captioner:
            out = await _resolve_scene(db, TEMPLATE, REPO, final=True)

        captioner.assert_not_called()
        assert out == "k3llydw, a woman in a red dress on a sofa, she grips the edge of the sofa"

    @pytest.mark.asyncio
    async def test_describing_a_repo_image_saves_it_for_next_time(self, db):
        with patch("app.routes.segments.s3.download_bytes", return_value=b"png"), \
             patch("app.routes.segments.caption_image_bytes",
                   new=AsyncMock(return_value=("a woman on a sofa", "an instruction"))):
            out = await _resolve_scene(db, TEMPLATE, REPO, final=True)

        assert "a woman on a sofa" in out
        meta = await db.get(ImageMeta, REPO)
        assert meta is not None
        assert meta.scene_description == "a woman on a sofa"
        assert meta.scene_instruction == "an instruction"

    @pytest.mark.asyncio
    async def test_a_generated_frame_is_described_but_never_stored(self, db):
        """Every continuation is this case. It still gets a description; it just does not
        get a row, because a row would put it in the Image Repo."""
        with patch("app.routes.segments.s3.download_bytes", return_value=b"png"), \
             patch("app.routes.segments.caption_image_bytes",
                   new=AsyncMock(return_value=("a woman by a pool", "an instruction"))):
            out = await _resolve_scene(db, TEMPLATE, GENERATED, final=True)

        assert "a woman by a pool" in out
        assert await db.get(ImageMeta, GENERATED) is None

    @pytest.mark.asyncio
    async def test_a_row_with_tags_but_no_description_still_describes(self, db):
        """A tagged image from before this existed. An empty description is not a
        description, and must not read as "already done"."""
        db.add(ImageMeta(path=REPO, tags="Kelly"))
        await db.flush()

        with patch("app.routes.segments.s3.download_bytes", return_value=b"png"), \
             patch("app.routes.segments.caption_image_bytes",
                   new=AsyncMock(return_value=("a woman on a sofa", "an instruction"))):
            out = await _resolve_scene(db, TEMPLATE, REPO, final=True)

        assert "a woman on a sofa" in out
        meta = await db.get(ImageMeta, REPO)
        assert meta.tags == "Kelly", "describing must not disturb the tags"

    @pytest.mark.asyncio
    async def test_a_filled_region_needs_no_captioner_at_all(self, db):
        """The words are already there. Resolving is just taking the markers off."""
        filled = "k3llydw, <scene>a woman on a sofa</scene>, she grips"
        with patch("app.routes.segments.caption_image_bytes", new=AsyncMock()) as captioner:
            out = await _resolve_scene(db, filled, REPO, final=True)

        captioner.assert_not_called()
        assert out == "k3llydw, a woman on a sofa, she grips"

    @pytest.mark.asyncio
    async def test_a_captioner_that_is_down_still_drops_the_placeholder(self, db):
        """Unchanged, and the reason it matters is unchanged: a literal <SCENE> is garbage
        tokens, while dropping it leaves a valid if generic prompt."""
        with patch("app.routes.segments.s3.download_bytes", side_effect=RuntimeError("down")):
            out = await _resolve_scene(db, TEMPLATE, REPO, final=True)

        assert SCENE_PLACEHOLDER not in out
        assert await db.get(ImageMeta, REPO) is None


class TestSceneAndWildcardsDoNotCollide:
    """A filled scene must be invisible to the wildcard resolver.

    <SCENE> is filled AFTER wildcards precisely so the caption is never re-scanned — a
    description is model output, and one containing <color> would otherwise have a random
    option spliced into it. The console filling the region before it sends (console#427)
    moved the caption into the prompt BEFORE the API sees it, so that ordering no longer
    protects anything on its own and the region has to be held out of reach explicitly.
    """

    @staticmethod
    async def _wildcard(db, name, options):
        from app.models import Wildcard

        db.add(Wildcard(name=name, options=options))
        await db.flush()

    @pytest.mark.asyncio
    async def test_a_caption_containing_a_wildcard_name_is_left_alone(self, db):
        from app.routes.segments import _resolve_wildcards_outside_scene

        await self._wildcard(db, "color", ["scarlet"])
        prompt = "k3llydw, <scene>a woman in a <color> dress</scene>, she grips"

        resolved, _ = await _resolve_wildcards_outside_scene(db, prompt)

        assert resolved == "k3llydw, a woman in a <color> dress, she grips"
        assert "scarlet" not in resolved, "the caption was scanned for wildcards"

    @pytest.mark.asyncio
    async def test_a_wildcard_OUTSIDE_the_scene_still_resolves(self, db):
        """The masking must not cost the feature it is protecting."""
        from app.routes.segments import _resolve_wildcards_outside_scene

        await self._wildcard(db, "color", ["scarlet"])
        prompt = "k3llydw, <scene>a woman on a sofa</scene>, a <color> light"

        resolved, _ = await _resolve_wildcards_outside_scene(db, prompt)

        assert resolved == "k3llydw, a woman on a sofa, a scarlet light"

    @pytest.mark.asyncio
    async def test_the_markers_do_not_count_as_wildcards(self, db):
        """`<scene>` and `</scene>` both match the resolver's pattern, as "scene" and
        "/scene". A match that substitutes nothing still sets the template, which is the
        record of "this prompt had wildcards in it"."""
        from app.routes.segments import _resolve_wildcards_outside_scene

        resolved, template = await _resolve_wildcards_outside_scene(
            db, "k3llydw, <scene>a woman on a sofa</scene>, she grips")

        assert resolved == "k3llydw, a woman on a sofa, she grips"
        assert template is None, "the markers were counted as wildcards"

    @pytest.mark.asyncio
    async def test_an_unfilled_placeholder_still_reaches_the_resolver_untouched(self, db):
        """<SCENE> is not a region and must survive to be described later."""
        from app.routes.segments import _resolve_wildcards_outside_scene

        resolved, _ = await _resolve_wildcards_outside_scene(db, TEMPLATE)
        assert SCENE_PLACEHOLDER in resolved

    @pytest.mark.asyncio
    async def test_the_template_records_the_prompt_with_its_wildcards_intact(self, db):
        from app.routes.segments import _resolve_wildcards_outside_scene

        await self._wildcard(db, "color", ["scarlet"])
        prompt = "k3llydw, <scene>a woman on a sofa</scene>, a <color> light"

        _, template = await _resolve_wildcards_outside_scene(db, prompt)

        # The scene comes back unwrapped -- the markers are a console affordance and mean
        # nothing to a stored segment -- while <color> is still there to be re-drawn.
        assert template == "k3llydw, a woman on a sofa, a <color> light"

    @pytest.mark.asyncio
    async def test_two_regions_keep_their_own_captions(self, db):
        from app.routes.segments import _resolve_wildcards_outside_scene

        resolved, _ = await _resolve_wildcards_outside_scene(
            db, "<scene>first</scene> then <scene>second</scene>")
        assert resolved == "first then second"
