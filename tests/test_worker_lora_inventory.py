"""The LoRA inventory a worker reports through its heartbeat (daemon#165).

A worker is the only thing that can see its own LoRA directory, so this is reported, not
derived. The API's job is to store it without corrupting it — and the one way to corrupt it
is to overwrite a good inventory with nothing.
"""
from app.schemas.workers import WorkerHeartbeat


def test_an_older_daemon_still_heartbeats():
    """`loras` must be optional.

    On upgrade day every worker is running the previous daemon. If this field were required
    they would all 422 on heartbeat and drop out of the pool — the fleet going offline
    because the API learned a new field.
    """
    hb = WorkerHeartbeat(comfyui_running=True)
    assert hb.loras is None


def test_an_inventory_round_trips_intact():
    inv = {
        "synced_at": "2026-09-02T15:00:13+00:00",
        "dir": "/workspace/models/loras",
        "items": [
            {"name": "k3lly2026_v2.safetensors", "kind": "character", "state": "current"},
            {"name": "sfbehind.safetensors", "kind": "content", "state": "deferred"},
        ],
    }
    hb = WorkerHeartbeat(comfyui_running=True, loras=inv)
    assert hb.loras == inv
    assert hb.loras["items"][1]["state"] == "deferred"


def test_omitting_loras_must_not_erase_a_stored_one():
    """The route writes only when the field is present, unlike a1111 which assigns
    unconditionally.

    An older daemon omits `loras` on every heartbeat. Assigning None would blank a good
    inventory seconds after a newer worker reported it, and the page would flicker between
    a real answer and nothing with no explanation.
    """
    import inspect

    from app.routes import workers as mod

    src = inspect.getsource(mod.heartbeat)
    assert "if body.loras is not None:" in src, (
        "loras must be written conditionally, or an older daemon erases it each heartbeat"
    )
