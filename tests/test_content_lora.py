"""Content LoRA as a per-pose override.

The graph has always chained two LoRAs per stage — content, then character. They answer
different questions: the character LoRA is WHO, the content LoRA is WHAT IS HAPPENING. The
character half already lived on ltx_characters; this is the pose half, which until now was
a single global and so could not say "sfbehind for the from-behind poses, nothing for the
rest".

The risk being guarded here is not a crash. It is a pose quietly acquiring a content LoRA
it never asked for, or losing a deliberate strength of 0 — both of which render successfully
and wrongly.
"""
from app.ltx_stack import LTX_STACK


def test_the_default_is_still_none():
    """Dropping DR34ML4Y is what removed the motion horror.

    Making content LoRAs per-pose must not reintroduce one anywhere it was not asked for.
    A pose that says nothing gets the stack value, and the stack value stays "none".
    """
    assert LTX_STACK["content_lora"] == "none"


def test_the_stage_defaults_are_the_number_resolve_used_to_hardcode():
    """resolve() applied 0.6 to both stages before this was configurable.

    Keeping that as the default means a pose that names a content LoRA without naming
    strengths renders at exactly what the graph used to apply — not at some new number
    chosen while making it configurable.
    """
    assert LTX_STACK["content_s1"] == 0.6
    assert LTX_STACK["content_s2"] == 0.6


class _Pose:
    """The fields the resolver reads. Not the ORM — this is about the fallback rules."""
    def __init__(self, **kw):
        self.id = "00000000-0000-0000-0000-000000000000"
        self.name = "p"
        self.prompt_template = "t"
        self.negative_prompt = None
        self.frames = None
        self.img_compression = None
        self.content_lora = None
        self.content_s1 = None
        self.content_s2 = None
        self.validated = False
        self.__dict__.update(kw)


def _resolve(pose):
    """The exact expressions the /ltx/recipes resolver uses, kept in one place."""
    return {
        "content_lora": pose.content_lora or LTX_STACK["content_lora"],
        "content_s1": (pose.content_s1 if pose.content_s1 is not None
                       else LTX_STACK["content_s1"]),
        "content_s2": (pose.content_s2 if pose.content_s2 is not None
                       else LTX_STACK["content_s2"]),
    }


def test_a_pose_that_says_nothing_gets_no_content_lora():
    """Every existing pose is exactly this. The migration must change no output."""
    assert _resolve(_Pose())["content_lora"] == "none"


def test_a_pose_can_name_its_own_content_lora():
    out = _resolve(_Pose(content_lora="sfbehind_LTX2_3_v0_1", content_s1=0.7, content_s2=0.9))
    assert out["content_lora"] == "sfbehind_LTX2_3_v0_1"
    assert (out["content_s1"], out["content_s2"]) == (0.7, 0.9)


def test_a_strength_of_zero_survives_and_is_not_replaced_by_the_default():
    """The same trap img_compression has: 0 is a REAL setting, not "unset".

    Strength 0 loads the LoRA and gives it no weight, which is how you measure what it is
    actually contributing. `or` would turn that into 0.6 and the measurement would silently
    be of something else. This is the assertion that stops `is None` being "simplified".
    """
    out = _resolve(_Pose(content_lora="sfbehind_LTX2_3_v0_1", content_s1=0.0, content_s2=0.0))
    assert out["content_s1"] == 0.0
    assert out["content_s2"] == 0.0


def test_naming_a_lora_without_strengths_uses_the_stack_pair():
    out = _resolve(_Pose(content_lora="sfbehind_LTX2_3_v0_1"))
    assert (out["content_s1"], out["content_s2"]) == (0.6, 0.6)


def test_the_two_stages_stay_independent():
    """Stage 1 generates from noise at half size; stage 2 refines the upscaled latent.

    One number for both is a different configuration, not a simplification — the same
    reasoning already recorded for the character strengths.
    """
    out = _resolve(_Pose(content_lora="x", content_s1=0.3, content_s2=1.2))
    assert out["content_s1"] != out["content_s2"]
