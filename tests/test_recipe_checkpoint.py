"""Per-pose base model (console#404).

`checkpoint` was a global, so every render used one base model and two could not be
compared. Four sit on the 3090 and only one was reachable.

THE RISK IS ACCEPTED, NOT ABSENT. Character LoRAs were trained against sulphur. Against
another base a LoRA whose keys do not line up fuses NOTHING and says nothing about it — the
engine's own lora_coverage() docstring records that there is no error, no warning and no
log line. The render comes back as the base model with none of the character in it.

Comparing base models is a legitimate reason to accept that. What must not happen is the
comparison being unreadable, which is why the engine now measures fusion against the
checkpoint the job actually renders on rather than an env var.
"""
from app.ltx_stack import LTX_STACK


class _Pose:
    def __init__(self, **kw):
        self.checkpoint = None
        self.__dict__.update(kw)


def _resolve(pose):
    """The expression the /ltx/recipes resolver uses."""
    return pose.checkpoint or LTX_STACK["checkpoint"]


def test_a_pose_that_says_nothing_still_renders_on_sulphur():
    """Every existing pose. The migration must change no output."""
    assert _resolve(_Pose()) == "sulphur_dev_bf16"


def test_a_pose_can_name_its_own_base_model():
    """The point of the change: sulphur vs 10Eros on one pose and one seed."""
    assert _resolve(_Pose(checkpoint="10Eros_v1.5_bf16")) == "10Eros_v1.5_bf16"


def test_clearing_it_falls_back_rather_than_rendering_on_an_empty_name():
    """"" reaches the resolver when a user clears the field. It must mean "use the stack",
    not "load a checkpoint called nothing"."""
    assert _resolve(_Pose(checkpoint="")) == "sulphur_dev_bf16"


def test_the_stack_default_is_unchanged():
    """This is what every validated result to date was produced on. If it moves, the whole
    rated history stops being comparable to anything new."""
    assert LTX_STACK["checkpoint"] == "sulphur_dev_bf16"
