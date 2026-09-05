"""The artifact kinds a worker says it can fetch, from heartbeat to response (console#422).

The claim gate reads this to decide that a LoRA the worker has never seen is not a reason to
refuse it work. Two ways it goes wrong, both silent:

  * a required field would 422 every older daemon off the pool on upgrade day
  * a field missing from the RESPONSE model reads as NULL from outside, whatever is stored

The second one actually happened: the column and the heartbeat shipped together and the
response schema did not, so a daemon that was reporting ["lora"] correctly looked like one
that had never reported at all.
"""
from app.schemas.workers import WorkerHeartbeat, WorkerResponse


def test_an_older_daemon_still_heartbeats():
    """Optional, like every field added before it. A required one takes the fleet offline
    because the API learned a new word."""
    assert WorkerHeartbeat(comfyui_running=True).fetchable_kinds is None


def test_what_the_worker_reports_round_trips():
    hb = WorkerHeartbeat(comfyui_running=True, fetchable_kinds=["lora"])
    assert hb.fetchable_kinds == ["lora"]


def test_omitting_it_must_not_erase_a_stored_value():
    """The route writes only when the field is present. Assigning None on every older-daemon
    heartbeat would blank what a newer one just reported."""
    import inspect

    from app.routes import workers as mod

    src = inspect.getsource(mod)
    assert "if body.fetchable_kinds is not None:" in src


def test_the_response_reports_it():
    """Pydantic drops what the response model does not name, silently, which makes a stored
    value and an unreported one indistinguishable from outside — and the gate's behaviour
    unexplainable from the API alone."""
    assert "fetchable_kinds" in WorkerResponse.model_fields


def test_the_response_survives_a_worker_that_never_reported():
    """NULL is a real state here: never reported, which the gate reads as "fetches nothing"."""
    assert WorkerResponse.model_fields["fetchable_kinds"].default is None
