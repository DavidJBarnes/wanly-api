"""Content LoRAs as a per-pose LIST (console#395, extended by console#410).

The graph has always chained two LoRAs per stage — content, then character. They answer
different questions: the character LoRA is WHO, the content LoRA is WHAT IS HAPPENING. The
character half lives on ltx_characters; this is the pose half.

They STACK, because motion, act and framing are separable. Order is part of the
configuration, not incidental.

The risk being guarded here is not a crash. It is a pose quietly acquiring a LoRA it never
asked for, or losing a deliberate strength of 0 — both of which render successfully and
wrongly.
"""
from app.joycaption import CAPTION_STYLES  # noqa: F401  (import guard: app package loads)
from app.ltx_stack import LTX_STACK
from app.schemas.ltx import ContentLora, LtxRecipeCreate, LtxRecipeUpdate


def test_the_stack_default_is_still_none():
    """Dropping DR34ML4Y is what removed the motion horror. Making content LoRAs per-pose,
    and then stackable, must not reintroduce one anywhere it was not asked for. A pose that
    says nothing gets an empty list."""
    assert LTX_STACK["content_lora"] == "none"


def test_the_stage_defaults_are_the_number_resolve_used_to_hardcode():
    """resolve() applied 0.6 to both stages before any of this was configurable. Keeping it
    as the per-entry default means naming a LoRA and touching nothing renders it at exactly
    what the graph already applied."""
    assert LTX_STACK["content_s1"] == 0.6
    assert LTX_STACK["content_s2"] == 0.6
    assert ContentLora(name="x").s1 == 0.6
    assert ContentLora(name="x").s2 == 0.6


class _Pose:
    """The field the resolver reads. Not the ORM — this is about the resolution rule."""
    def __init__(self, content_loras=None):
        self.content_loras = content_loras


def _resolve(pose):
    """The expression the /ltx/recipes resolver uses."""
    return {"content_loras": pose.content_loras or []}


def test_a_pose_that_says_nothing_gets_no_content_lora():
    """Every existing pose. The migration must change no output."""
    assert _resolve(_Pose())["content_loras"] == []


def test_a_pose_can_stack_several_in_order():
    """The point of console#410. Order is preserved because it is the application order."""
    loras = [{"name": "sfbehind", "s1": 0.3, "s2": 0.9},
             {"name": "deepthroat", "s1": 0.5, "s2": 1.0}]
    assert [c["name"] for c in _resolve(_Pose(loras))["content_loras"]] == ["sfbehind", "deepthroat"]


def test_a_strength_of_zero_survives_the_schema():
    """0 is a REAL setting: the LoRA loads and contributes nothing, which is how you measure
    what it was contributing. Nothing may quietly promote it to the 0.6 default."""
    c = ContentLora(name="x", s1=0.0, s2=0.0)
    assert c.s1 == 0.0 and c.s2 == 0.0


def test_the_api_bound_is_not_looser_than_the_engines():
    """The engine bounds each strength at 2.0 (engine/app.py ContentLora).

    A looser bound here accepts a value the console stores happily and the engine then
    rejects with a 422 — ten minutes into a claimed segment, not in the dialog. Whichever
    end is stricter is the real limit, and this is the one that can drift unnoticed.
    """
    ENGINE_MAX = 2.0
    for field in ("s1", "s2"):
        le = next(m.le for m in ContentLora.model_fields[field].metadata if hasattr(m, "le"))
        assert le <= ENGINE_MAX, f"ContentLora.{field} allows {le} > engine {ENGINE_MAX}"


def test_the_list_is_capped_to_match_the_engine():
    """LtxRequest.loras caps at 4; the content chain should not exceed it either. Four LoRAs
    on one chain is already a lot of competition for the same weights."""
    for model in (LtxRecipeCreate, LtxRecipeUpdate):
        meta = model.model_fields["content_loras"].metadata
        mx = next((m.max_length for m in meta if hasattr(m, "max_length")), None)
        assert mx == 4, f"{model.__name__}.content_loras is not capped at 4"
