"""What a worker IS, and the two places that must read it (#269).

wanly-services is a real, long-lived, GPU-holding process that can never render. Adding it to
the fleet is only safe if two things key on `kind`, and each has a failure that is silent:

  * THE CLAIM GATE. A freshly registered service has `checkpoints` NULL, and `_model_gate`
    deliberately does not filter a worker that has never reported checkpoints -- so without a
    kind check it would be offered every pending segment. Nothing breaks today only because
    wanly-services has no daemon and never polls, which is the safety being an accident of it
    not asking rather than a decision that it must not be asked.

  * QUEUE HEALTH. `stalled` means "work queued and nobody who can take it". A service counted
    as live means the alarm can never fire again -- and it would go quiet at exactly the
    moment it matters: every render worker down, work queued, one healthy captioner holding
    the count above zero.
"""
import inspect

from sqlalchemy import false as sa_false

from app.enums import WorkerKind, WorkerStatus
from app.models import Worker
from app.queue_health import COUNTED_KINDS, assess
from app.routes.segments import _model_gate
from app.schemas.workers import WorkerHeartbeat, WorkerRegister, WorkerResponse


def _worker(**kw):
    w = Worker(friendly_name="w", hostname="h", ip_address="1.2.3.4")
    w.kind = kw.pop("kind", WorkerKind.RENDER)
    for k, v in kw.items():
        setattr(w, k, v)
    return w


class TestTheClaimGate:
    def test_a_service_is_offered_nothing(self):
        gate = _model_gate(_worker(kind=WorkerKind.SERVICE, checkpoints=None))
        assert len(gate) == 1
        assert str(gate[0]) == str(sa_false()), "a service must be filtered out, not unfiltered"

    def test_the_kind_check_beats_the_never_reported_exemption(self):
        """The exemption returns an EMPTY tuple, meaning "no restriction". A service reaching
        it is offered the whole queue, so the kind check has to come first — this asserts the
        ordering, which is the entire correctness of the fix."""
        src = inspect.getsource(_model_gate)
        kind_at = src.index("worker.kind != WorkerKind.RENDER")
        exempt_at = src.index("worker.checkpoints is None")
        assert kind_at < exempt_at, \
            "the never-reported exemption runs first and waves the service through"

    def test_a_service_that_somehow_reported_checkpoints_is_still_refused(self):
        """Kind decides, not inventory. A service sharing a models mount could plausibly see
        checkpoints; that must not make it claimable."""
        gate = _model_gate(_worker(kind=WorkerKind.SERVICE, checkpoints=["10Eros_v1.5_bf16"]))
        assert str(gate[0]) == str(sa_false())

    def test_a_render_worker_is_completely_unaffected(self):
        """The whole fleet must keep behaving exactly as before."""
        assert _model_gate(_worker(checkpoints=None)) == ()
        assert _model_gate(None) == ()

    def test_a_render_worker_with_inventory_still_gates_on_it(self):
        gate = _model_gate(_worker(checkpoints=["10Eros_v1.5_bf16"], fetchable_kinds=[]))
        assert len(gate) == 1
        assert str(gate[0]) != str(sa_false())


class TestQueueHealth:
    def test_only_render_workers_are_counted(self):
        assert COUNTED_KINDS == frozenset({"render"})

    def test_the_query_filters_on_kind(self):
        """Counting every row is what made this a regression rather than a preference."""
        from app.routes import workers as mod
        src = inspect.getsource(mod.queue_health)
        assert "Worker.kind.in_(COUNTED_KINDS)" in src

    def test_stalled_still_fires_when_only_a_service_is_alive(self):
        """The scenario the filter exists for: the captioner is fine, every render worker is
        gone, and there is work. Because queue_health filters before calling assess, the
        service simply is not in this list."""
        h = assess(pending_segments=4, worker_statuses=["offline"])
        assert h.stalled is True
        assert h.live_workers == 0


class TestRegistration:
    def test_an_older_daemon_registers_as_render(self):
        """No daemon in the fleet sends `kind`. Every one of them must keep meaning what it
        meant, and the safe default is the one that keeps them working."""
        body = WorkerRegister(friendly_name="3090.zero", hostname="h", ip_address="1.2.3.4")
        assert body.kind == WorkerKind.RENDER
        assert body.provides is None

    def test_a_service_declares_itself(self):
        body = WorkerRegister(friendly_name="2070.zero/services", hostname="h",
                              ip_address="1.2.3.4", kind=WorkerKind.SERVICE,
                              provides=["joycaption"])
        assert body.kind == WorkerKind.SERVICE
        assert body.provides == ["joycaption"]

    def test_re_registering_can_change_what_a_box_is(self):
        """A host could stop running services and start running an engine. The row should
        follow reality rather than the first thing it ever saw."""
        from app.routes import workers as mod
        src = inspect.getsource(mod.register_worker)
        assert "worker.kind = body.kind" in src

    def test_provides_is_a_conditional_write_on_register(self):
        from app.routes import workers as mod
        src = inspect.getsource(mod.register_worker)
        assert "if body.provides is not None:" in src


class TestHeartbeat:
    def test_an_older_daemon_still_heartbeats(self):
        hb = WorkerHeartbeat(comfyui_running=True)
        assert hb.provides is None and hb.status is None

    def test_provides_is_a_conditional_write(self):
        """Assigning None on every older-daemon heartbeat would blank what a newer one just
        reported — the mistake loras and checkpoints each had to fix."""
        from app.routes import workers as mod
        src = inspect.getsource(mod.heartbeat)
        assert "if body.provides is not None:" in src

    def test_a_service_sets_its_own_status(self):
        from app.routes import workers as mod
        src = inspect.getsource(mod.heartbeat)
        assert "if body.status is not None:" in src

    def test_a_render_workers_status_is_never_taken_from_the_body(self):
        """Otherwise a daemon could talk itself into looking idle, which is precisely the
        confusion wanly-gpu-docker#80 is open about."""
        from app.routes import workers as mod
        src = inspect.getsource(mod.heartbeat)
        service_branch = src.index("if worker.kind != WorkerKind.RENDER:")
        status_write = src.index("if body.status is not None:")
        else_branch = src.index("    else:\n        if worker.status == \"offline\":")
        assert service_branch < status_write < else_branch, \
            "body.status is honoured outside the service branch"


class TestTheResponse:
    def test_it_reports_both(self):
        """Pydantic silently drops what the response model does not name, which is how a
        correctly stored fetchable_kinds once looked like a worker that had never reported."""
        assert "kind" in WorkerResponse.model_fields
        assert "provides" in WorkerResponse.model_fields


class TestTheStatusVocabulary:
    def test_a_service_has_words_of_its_own(self):
        """Reusing online-idle was the cheap wrong answer: it already means both "waiting for
        work" and "cannot do work", and that ambiguity hid a dead ComfyUI for 33 minutes."""
        assert WorkerStatus.ONLINE == "online"
        assert WorkerStatus.DEGRADED == "degraded"

    def test_service_statuses_are_not_live_for_the_queue(self):
        """Belt and braces behind the kind filter: even if a service row reached assess(), its
        status is not one that counts."""
        from app.queue_health import LIVE_STATUSES
        assert WorkerStatus.ONLINE not in LIVE_STATUSES
        assert WorkerStatus.DEGRADED not in LIVE_STATUSES
