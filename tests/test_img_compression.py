"""img_compression: a per-pose override that must survive being zero.

0 is a REAL setting — comfy_extras/nodes_lt.py `preprocess` returns the image untouched at
crf 0 — so every hop has to use `is None` rather than truthiness. A falsy check silently
replaces "skip the encode" with the stack default, which is a different render that looks
plausible and is not what the pose asked for.
"""
import pytest

from app.ltx_stack import LTX_STACK


def resolve(pose_value):
    """Mirrors the book's resolution, which is the rule under test."""
    return pose_value if pose_value is not None else LTX_STACK["img_compression"]


def test_the_stack_carries_the_value_every_rated_result_used():
    assert LTX_STACK["img_compression"] == 18


def test_a_pose_with_no_override_uses_the_stack():
    assert resolve(None) == 18


def test_a_pose_override_wins():
    assert resolve(4) == 4


def test_zero_is_honoured_and_not_treated_as_unset():
    # The bug this guards: `r.img_compression or LTX_STACK[...]` yields 18 here, so a pose
    # asking to skip the conditioning encode would quietly render at the default instead.
    assert resolve(0) == 0


@pytest.mark.parametrize("bad", [-1, 52, 100])
def test_out_of_range_is_rejected_by_the_schema(bad):
    from pydantic import ValidationError

    from app.schemas.ltx import LtxRecipeCreate

    with pytest.raises(ValidationError):
        LtxRecipeCreate(name="p", prompt_template="<TRIGGER>, x", img_compression=bad)


@pytest.mark.parametrize("ok", [0, 4, 18, 51])
def test_the_meaningful_crf_range_is_accepted(ok):
    from app.schemas.ltx import LtxRecipeCreate

    # 0-51 is x264's actual scale. The node accepts up to 100, but above 51 the number stops
    # meaning anything, so the schema is the tighter of the two on purpose.
    assert LtxRecipeCreate(name="p", prompt_template="<TRIGGER>, x",
                           img_compression=ok).img_compression == ok
