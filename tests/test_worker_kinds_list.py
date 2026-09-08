"""A worker may be several kinds at once (wanly-gpu-docker#83).

One container per GPU registers once as ["render", "trainer"]. `kind` stays the first of
those, render first whenever present, so every render gate and the queue-health count keep
reading one word; the trainer half is visible only through `kinds`, which is what the
training gate reads.
"""
import uuid

import pytest

from app.enums import WorkerKind, ordered_kinds, worker_can, worker_kinds
from app.models import Worker
from app.schemas.workers import WorkerRegister


def _worker(**kw):
    base = dict(id=uuid.uuid4(), friendly_name="3090.zero", hostname="h", ip_address="1.2.3.4",
                kind="render", kinds=None, status="online-idle")
    base.update(kw)
    return Worker(**base)


class TestTheOneReader:
    def test_a_row_from_before_the_column_is_its_one_kind(self):
        w = _worker(kind="trainer", kinds=None)
        assert worker_kinds(w) == ["trainer"]
        assert worker_can(w, WorkerKind.TRAINER) and not worker_can(w, WorkerKind.RENDER)

    def test_a_box_that_is_both_is_both(self):
        w = _worker(kind="render", kinds=["render", "trainer"])
        assert worker_can(w, WorkerKind.RENDER) and worker_can(w, WorkerKind.TRAINER)

    def test_render_is_always_first(self):
        """`kind` is kinds[0], and the render gates and queue-health read `kind`."""
        assert ordered_kinds(["trainer", "render"]) == ["render", "trainer"]
        assert ordered_kinds(["trainer", "trainer"]) == ["trainer"]
        assert ordered_kinds([WorkerKind.SERVICE]) == ["service"]


class TestRegistration:
    def test_an_old_daemon_sends_one_kind_and_that_is_the_list(self):
        body = WorkerRegister(friendly_name="x", hostname="h", ip_address="1.2.3.4")
        assert body.kinds is None and body.kind == WorkerKind.RENDER

    def test_kinds_are_validated(self):
        with pytest.raises(ValueError):
            WorkerRegister(friendly_name="x", hostname="h", ip_address="1.2.3.4", kinds=["renderer"])

    def test_the_route_derives_kind_from_kinds_render_first(self):
        import inspect
        from app.routes import workers as mod
        src = inspect.getsource(mod.register_worker)
        assert "kinds = ordered_kinds(body.kinds or [body.kind])" in src
        assert "worker.kind = kinds[0]" in src and "worker.kinds = kinds" in src

    async def test_a_box_registering_as_both_gets_the_render_vocabulary(self, db):
        from app.routes.workers import register_worker
        w = await register_worker(WorkerRegister(
            friendly_name="3090.zero", hostname="h", ip_address="1.2.3.4",
            kinds=[WorkerKind.TRAINER, WorkerKind.RENDER]), db=db)
        assert w.kind == "render" and w.kinds == ["render", "trainer"]
        assert w.status == "online-idle"


class TestTheTrainingGate:
    def test_reads_kinds_not_kind(self):
        import inspect
        from app.routes import training as mod
        src = inspect.getsource(mod.claim_next_training_job)
        assert "worker_can(worker, WorkerKind.TRAINER)" in src
        assert "worker.kind != WorkerKind.TRAINER" not in src

    def test_the_render_gate_still_reads_kind_because_render_is_first(self):
        """No change needed there: kind == render exactly when render is among the kinds."""
        import inspect
        from app.routes import segments as mod
        assert "worker.kind != WorkerKind.RENDER" in inspect.getsource(mod._model_gate)
