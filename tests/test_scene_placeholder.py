"""<SCENE>: filling a pose's static half from the frame it actually starts on.

console#405. Recipes are start-frame-agnostic, so their static half is a guess about an
image they have never seen — a validated pose says "a woman kneeling in front of a nude man"
while the frame is a clothed woman on a sofa. Every test here is about that substitution
being safe, because the failure modes are all silent: a wrong caption renders fine and looks
plausible.
"""
import pytest

from app.joycaption import CAPTION_STYLES, DEFAULT_STYLE, image_key, instruction_for
from app.routes.segments import SCENE_PLACEHOLDER, _drop_scene


class TestStyles:
    def test_every_style_but_raw_asks_for_gaze_and_expression(self):
        """"looking at the viewer" is load-bearing in these prompts, and a plain
        "describe this image" omits it. Asking explicitly is what separated a usable caption
        from a nearly-usable one on a real frame."""
        for name, prompt in CAPTION_STYLES.items():
            if name == "raw":
                continue
            assert "looking" in prompt, f"{name} does not ask where the subject is looking"

    def test_every_style_but_raw_suppresses_text_and_framing(self):
        """The model reports overlay text and picture frames unless told not to — on a real
        frame it read cursive in the corner and confabulated it into a person's name. True of
        the image, garbage in a render prompt."""
        for name, prompt in CAPTION_STYLES.items():
            if name == "raw":
                continue
            assert "text or writing" in prompt, f"{name} does not suppress overlay text"
            assert "camera" in prompt, f"{name} does not suppress camera talk"

    def test_a_custom_instruction_always_wins(self):
        """The presets encode what tested well; the person tuning prompts knows better."""
        assert instruction_for("terse", "my own words") == "my own words"
        assert instruction_for("rich", "   ") == CAPTION_STYLES["rich"]     # blank is not custom

    def test_an_unknown_style_falls_back_rather_than_raising(self):
        """A style written straight into app_settings, or one dropped in a later release,
        must not take the settings page down with it."""
        assert instruction_for("nonsense") == CAPTION_STYLES[DEFAULT_STYLE]


class TestCacheKey:
    def test_the_same_image_and_instruction_give_the_same_key(self):
        """A retry must reproduce the caption its first attempt used. Without that, retrying
        a failed segment re-captions, gets different words, and renders something different
        from what failed — which quietly breaks what "retry" means."""
        assert image_key(b"same-bytes", "same") == image_key(b"same-bytes", "same")

    def test_changing_the_instruction_changes_the_key(self):
        """Switching style should produce a new caption, not serve the old one."""
        assert image_key(b"img", CAPTION_STYLES["terse"]) != image_key(b"img", CAPTION_STYLES["rich"])

    def test_it_keys_on_content_not_path(self):
        """The same frame is referenced from several places; two paths to one image must
        share a caption."""
        assert image_key(b"identical", "i") == image_key(b"identical", "i")
        assert image_key(b"different", "i") != image_key(b"identical", "i")


class TestDropping:
    """When the captioner is down the placeholder is DROPPED, not left literal.

    This inverts _resolve_trigger, which leaves its placeholder deliberately: dropping the
    token that anchors the character LoRA is worse and harder to notice. Here the reasoning
    flips — a literal "<SCENE>" is garbage tokens to the encoder, while dropping leaves a
    valid if generic prompt, which is what every pose had before this existed.
    """

    def test_the_placeholder_never_survives(self):
        assert SCENE_PLACEHOLDER not in _drop_scene(f"trig, {SCENE_PLACEHOLDER}, she grips")

    def test_no_dangling_empty_clause_is_left(self):
        """"trig, , she grips" reaches the encoder as a stray empty clause."""
        out = _drop_scene(f"trig, {SCENE_PLACEHOLDER}, she grips his hand")
        assert ", ," not in out
        assert out == "trig, she grips his hand"

    @pytest.mark.parametrize("prompt,expected", [
        (f"{SCENE_PLACEHOLDER}, she moves", "she moves"),
        (f"trig, {SCENE_PLACEHOLDER}", "trig"),
        (f"a {SCENE_PLACEHOLDER} b", "a b"),
    ])
    def test_it_tidies_in_every_position(self, prompt, expected):
        assert _drop_scene(prompt) == expected

    def test_a_prompt_without_the_placeholder_is_untouched(self):
        """Every pose today. This must be a no-op for them."""
        p = "k3lly2026, a woman kneeling, she grips his hand"
        assert _drop_scene(p) == p


class TestReservation:
    def test_scene_is_reserved_against_wildcards(self):
        """SCENE resolves AFTER wildcards so the caption is never wildcard-expanded — which
        means the wildcard resolver sees the placeholder FIRST. A wildcard named SCENE would
        substitute a random option before the captioner is called, and the render would look
        plausible and be wrong. For TRIGGER the reservation is a second belt; for SCENE it is
        the only guard.
        """
        from app.routes.wildcards import RESERVED_WILDCARD_NAMES
        assert "SCENE" in RESERVED_WILDCARD_NAMES
        assert "TRIGGER" in RESERVED_WILDCARD_NAMES


class TestDeferral:
    """A continuation is routinely created BEFORE the segment it follows has rendered.

    Its start frame is that segment's last frame, which does not exist yet. Resolving at
    submit would therefore find nothing to describe — and dropping the placeholder there
    would be unrecoverable, because no later step could tell a description was ever wanted.

    Continuations are the case this feature helps MOST: they condition on a generated frame
    nobody has ever described, using recipe wording written before that frame existed. So
    losing them silently is the worst outcome available, and these tests pin the deferral.
    """

    @pytest.mark.asyncio
    async def test_submit_leaves_the_placeholder_when_there_is_no_image_yet(self):
        from app.routes.segments import _resolve_scene
        prompt = f"trig, {SCENE_PLACEHOLDER}, she moves"
        out = await _resolve_scene(None, prompt, None, final=False)
        assert out == prompt, "submit must defer, not drop"

    @pytest.mark.asyncio
    async def test_claim_drops_it_when_there_is_still_no_image(self):
        """Last responsible moment: a text-to-video segment, or a predecessor that produced
        no last frame. Shipping a literal <SCENE> to the encoder is the one outcome that is
        strictly worse than a generic prompt."""
        from app.routes.segments import _resolve_scene
        out = await _resolve_scene(None, f"trig, {SCENE_PLACEHOLDER}, she moves", None, final=True)
        assert SCENE_PLACEHOLDER not in out
        assert out == "trig, she moves"

    @pytest.mark.asyncio
    async def test_a_prompt_without_the_placeholder_never_calls_the_captioner(self):
        """Every pose today. The captioner must not be touched for them — this runs on the
        claim path, which every worker polls."""
        from app.routes.segments import _resolve_scene
        p = "k3lly2026, a woman kneeling, she grips his hand"
        assert await _resolve_scene(None, p, "s3://bucket/would-explode-if-fetched", final=True) == p
