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


class TestCheckpointListing:
    """What the dropdown offers (console#404).

    The union of what LIVE workers report. A checkpoint is a 46 GB file on a GPU box, so
    whether one is loadable is a fact about that box — and the engine binds to localhost,
    so workers report it through the heartbeat rather than the API discovering it.
    """

    def test_offline_workers_are_excluded_from_the_query(self):
        """Offering a checkpoint that exists only on a box which is not running produces a
        job nothing can claim — a queue that silently stops rather than an error."""
        import inspect
        from app.routes import ltx_recipes as mod
        src = inspect.getsource(mod.list_checkpoints)
        assert 'Worker.status != "offline"' in src

    def test_the_stack_default_is_always_offered(self):
        """With no worker online the dropdown must still contain the value every existing
        pose already renders on, or the field looks broken when the fleet is idle."""
        import inspect
        from app.routes import ltx_recipes as mod
        src = inspect.getsource(mod.list_checkpoints)
        assert 'LTX_STACK["checkpoint"]' in src

    def test_an_older_daemon_still_heartbeats(self):
        """checkpoints must be optional. Required, every worker on the previous daemon
        would 422 and drop out of the pool the moment this deployed."""
        from app.schemas.workers import WorkerHeartbeat
        assert WorkerHeartbeat(comfyui_running=True).checkpoints is None

    def test_omitting_it_must_not_erase_a_stored_list(self):
        """An older daemon omits the field on every heartbeat; assigning None would blank a
        good list seconds after a newer worker reported it."""
        import inspect
        from app.routes import workers as mod
        assert "if body.checkpoints is not None:" in inspect.getsource(mod.heartbeat)
