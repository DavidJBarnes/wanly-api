"""<MOTION>: filling a pose's motion half from the saved motion description (wanly-api#326→#335).

#326 writes the second caption paragraph — the frame read as the first frame of a ten-second
clip — and nothing consumed it. This is the render side: prompts embed <MOTION> the way they
embed <SCENE>, and resolution fills it.

The invariant the whole file protects is the one that separates this from <SCENE>:
**<MOTION> is cache-only.** It never triggers a caption call, at submit or at the claim — the
daemon's claim poll has a 10s HTTP timeout, a warm motion caption is ~50s, production's
captioner (MOTION_CAPTION_ENABLED=false) cannot answer the directional prompt at all, and the
paragraph's value is that a person read it in the lightbox. Every test that matters asserts
either "filled from the row" or "not called, dropped" — a live call in either direction fails
loudly here because the captioner is a mock that records.
"""
import pytest
from unittest.mock import AsyncMock, patch

from app.config import settings
from app.models import ImageMeta
from app.routes.segments import (
    MOTION_PLACEHOLDER,
    SCENE_PLACEHOLDER,
    _drop_motion,
    _resolve_scene,
)

REPO = f"s3://{settings.s3_images_bucket}/2026-09-05/00001.png"
MOTION = "she rocks back slowly, her hair swaying, his hands gripping her waist"
TEMPLATE = f"k3llydw, {SCENE_PLACEHOLDER}, {MOTION_PLACEHOLDER}"


class TestDropping:
    """Same inverted-from-<TRIGGER> rule as _drop_scene: a literal <MOTION> is garbage tokens
    to the text encoder; dropping it leaves the pose's own arc, which is what every pose had
    before this existed."""

    def test_the_placeholder_never_survives_a_drop(self):
        assert MOTION_PLACEHOLDER not in _drop_motion(f"trig, {MOTION_PLACEHOLDER}, she grips")

    def test_no_dangling_empty_clause_is_left(self):
        out = _drop_motion(f"trig, {MOTION_PLACEHOLDER}, she grips his hand")
        assert ", ," not in out
        assert out == "trig, she grips his hand"

    @pytest.mark.parametrize("prompt,expected", [
        (f"{MOTION_PLACEHOLDER}, she moves", "she moves"),
        (f"trig, {MOTION_PLACEHOLDER}", "trig"),
        (f"a {MOTION_PLACEHOLDER} b", "a b"),
    ])
    def test_it_tidies_in_every_position(self, prompt, expected):
        assert _drop_motion(prompt) == expected

    def test_a_prompt_without_the_placeholder_is_untouched(self):
        p = "k3lly2026, a woman kneeling, she grips his hand"
        assert _drop_motion(p) == p


class TestReservation:
    def test_motion_is_reserved_against_wildcards(self):
        """Same reasoning that makes SCENE's reservation the only guard there is: MOTION
        resolves after wildcards so the caption text is never expanded, which means the
        wildcard resolver sees the placeholder FIRST. A wildcard named MOTION would
        substitute a random option before resolution ever ran."""
        from app.routes.wildcards import RESERVED_WILDCARD_NAMES
        assert "MOTION" in RESERVED_WILDCARD_NAMES


class TestCacheOnly:
    """The rule #335 is named for: resolution reads the row and never the captioner."""

    @pytest.mark.asyncio
    async def test_a_saved_motion_description_fills_the_placeholder(self, db):
        db.add(ImageMeta(path=REPO, motion_description=MOTION))
        await db.flush()

        with patch("app.routes.segments.caption_image_bytes", new=AsyncMock()) as captioner:
            out = await _resolve_scene(db, f"trig, {MOTION_PLACEHOLDER}", REPO, final=True)

        captioner.assert_not_called()
        assert out == f"trig, {MOTION}"

    @pytest.mark.asyncio
    async def test_resolution_never_captions_even_when_the_frame_is_undescribed(self, db):
        """A frame with no motion row at the claim: drop, do not describe. This is the test
        that would fail if anyone ever "helpfully" wired a live motion caption into the claim
        path — a ~50 s warm call against the daemon's 10 s claim timeout, on a captioner
        production has disabled because it produces junk for this prompt."""
        with patch("app.routes.segments.caption_image_bytes", new=AsyncMock()) as captioner, \
             patch("app.routes.segments.s3.download_bytes") as s3get:
            out = await _resolve_scene(db, f"trig, {MOTION_PLACEHOLDER}", REPO, final=True)

        captioner.assert_not_called()
        s3get.assert_not_called()
        assert MOTION_PLACEHOLDER not in out
        assert out == "trig"

    @pytest.mark.asyncio
    async def test_resolution_never_saves_either(self, db):
        """Cache-only means write-only-by-the-console too: resolution must not mint a motion
        row, or an unreviewed caption becomes authoritative ground truth."""
        await _resolve_scene(db, f"trig, {MOTION_PLACEHOLDER}", REPO, final=True)
        meta = await db.get(ImageMeta, REPO)
        assert meta is None or not (meta.motion_description or "")


class TestDeferral:
    """A miss at submit SURVIVES: the Next Segment dialog (console#438) may describe the
    frame between creation and claim, and the claim is the last moment that can matter.
    Dropping at submit would be unrecoverable — the stored prompt would no longer show that
    a motion half was ever wanted."""

    @pytest.mark.asyncio
    async def test_submit_defers_a_miss_rather_than_dropping_it(self, db):
        with patch("app.routes.segments.caption_image_bytes", new=AsyncMock()) as captioner:
            out = await _resolve_scene(db, f"trig, {MOTION_PLACEHOLDER}", REPO, final=False)

        captioner.assert_not_called()
        assert out == f"trig, {MOTION_PLACEHOLDER}"

    @pytest.mark.asyncio
    async def test_submit_defers_when_there_is_no_image_at_all(self, db):
        """A continuation created before its predecessor rendered. Same rule as <SCENE>."""
        out = await _resolve_scene(db, f"trig, {MOTION_PLACEHOLDER}", None, final=False)
        assert out == f"trig, {MOTION_PLACEHOLDER}"

    @pytest.mark.asyncio
    async def test_the_claim_drops_it_when_the_miss_persisted(self, db):
        out = await _resolve_scene(db, f"trig, {MOTION_PLACEHOLDER}", REPO, final=True)
        assert out == "trig"


class TestBothHalves:
    """<SCENE> and <MOTION> resolve together: one ImageMeta read for the pair, a single
    fill pass, and the halves never reach into each other."""

    @pytest.mark.asyncio
    async def test_both_placeholders_fill_from_one_row(self, db):
        db.add(ImageMeta(path=REPO, scene_description="a woman on a sofa",
                         motion_description=MOTION))
        await db.flush()

        with patch("app.routes.segments.caption_image_bytes", new=AsyncMock()) as captioner:
            out = await _resolve_scene(db, TEMPLATE, REPO, final=True)

        captioner.assert_not_called()
        assert out == f"k3llydw, a woman on a sofa, {MOTION}"

    @pytest.mark.asyncio
    async def test_a_scene_miss_still_live_captions_while_motion_defers(self, db):
        """The halves keep their own failure rules: the scene asks the captioner like it
        always has, and the motion miss defers rather than riding along on that call."""
        with patch("app.routes.segments.s3.download_bytes", return_value=b"png"), \
             patch("app.routes.segments.caption_image_bytes",
                   new=AsyncMock(return_value=("a woman on a sofa", "an instruction"))):
            out = await _resolve_scene(db, TEMPLATE, REPO, final=False)

        assert out == f"k3llydw, a woman on a sofa, {MOTION_PLACEHOLDER}"

    @pytest.mark.asyncio
    async def test_a_filled_scene_does_not_rescan_as_motion(self, db):
        """Single-pass fill: words put in for one half are not re-read as the other half's
        placeholder. A static caption that happened to contain the literal token stays a
        curiosity — and at the claim the literal is scrubbed rather than rendered, because
        the encoder-never-sees-a-literal rule outranks caption verbatim."""
        db.add(ImageMeta(path=REPO, scene_description=f'a woman mouthing "{MOTION_PLACEHOLDER}"',
                         motion_description=MOTION))
        await db.flush()

        out = await _resolve_scene(db, TEMPLATE, REPO, final=True)

        assert MOTION_PLACEHOLDER not in out
        assert out.count(MOTION) == 1, "the motion text was substituted twice"
        assert "a woman mouthing" in out

    @pytest.mark.asyncio
    async def test_a_caption_embedded_literal_never_reaches_the_encoder(self, db):
        """The claim-path scrub: a literal token embedded in saved caption text (written
        before this existed, or pasted by hand) is removed at the last responsible moment."""
        db.add(ImageMeta(path=REPO, scene_description=f"scene about {MOTION_PLACEHOLDER} here",
                         motion_description=MOTION))
        await db.flush()

        out = await _resolve_scene(db, TEMPLATE, REPO, final=True)

        assert MOTION_PLACEHOLDER not in out
        assert SCENE_PLACEHOLDER not in out

    @pytest.mark.asyncio
    async def test_both_misses_at_the_claim_drop_without_a_dangling_clause(self, db):
        """A captioner that is down degrades the scene as it always did, and the motion miss
        joins the drop without leaving ", ," behind."""
        with patch("app.routes.segments.s3.download_bytes", side_effect=RuntimeError("down")):
            out = await _resolve_scene(db, f"trig, {SCENE_PLACEHOLDER}, {MOTION_PLACEHOLDER}, go",
                                       REPO, final=True)
        assert out == "trig, go"


class TestNoPlaceholdersUntouched:
    @pytest.mark.asyncio
    async def test_a_prompt_with_neither_token_touches_nothing(self, db):
        """Every pose today. This runs on the claim path, which every worker polls."""
        p = "k3lly2026, a woman kneeling, she grips his hand"
        out = await _resolve_scene(db, p, "s3://bucket/would-explode-if-fetched", final=True)
        assert out == p
