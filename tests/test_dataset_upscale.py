"""A dataset's "Upscale" (wanly-api#460): high-res copies of chosen stills, beside each one.

Pins what David asked for -- a COPY, saved to the set, the source kept -- and the guards it
shares with Fix small faces: one image per service call, an older service (no target_edge)
refused before anything is stored, a locked set refused, the write applied at the end.
"""
import base64
from contextlib import asynccontextmanager

import pytest

from app import dataset_upscale, face_size
from tests.test_dataset_face_size import SIZES, _Resp, _S3, _Service, _ds, _imgs, _wire


class _UpService(_Service):
    """The face-crop service with wanly-gpu-docker#214's target_edge, unless `ignores_target`
    (an older one: caps at 1024 and echoes nothing)."""

    def __init__(self, ignores_target=False, no_target_feature=False):
        super().__init__()
        self.ignores_target, self.no_target_feature = ignores_target, no_target_feature
        self.upscale_bodies: list[dict] = []

    def handle(self, method, path, body):
        if path == "/health":
            self.calls.append(path)
            feats = ["crop", "embed", "measure", "upscale", "pair"]
            if not self.no_target_feature:
                feats.append("upscale_target")
            return _Resp({"status": "ok", "features": feats})
        if path == "/upscale":
            self.calls.append(path)
            self.upscale_bodies.append(body)
            imgs = [base64.b64decode(b) for b in body["images"]]
            out = {"images": [{"upscaled": True, "b64": base64.b64encode(b"UP:" + b).decode(),
                               "width": 2048, "height": 1536} for b in imgs], "upscale": True}
            if not self.ignores_target:
                out["target_edge"] = body.get("target_edge")
            return _Resp(out)
        return super().handle(method, path, body)


@pytest.fixture
def shared_session(db, monkeypatch):
    from app.routes import datasets as mod

    @asynccontextmanager
    async def _session():
        yield db
    monkeypatch.setattr(mod, "async_session", _session)
    return db


def test_the_target_is_the_factor_of_the_long_edge_capped():
    assert dataset_upscale.target_edge(1024, 768, 2) == 2048
    assert dataset_upscale.target_edge(768, 1024, 4) == 4096
    assert dataset_upscale.target_edge(3000, 2000, 2) == 4096


@pytest.mark.asyncio
class TestUpscale:
    async def _world(self, db, monkeypatch, svc=None, prefix="u"):
        imgs = _imgs(3, prefix)
        a, b, big = imgs
        SIZES.update({a: (1024, 768, 300.0), b: (800, 1200, 300.0), big: (4096, 3000, 300.0)})
        # `a` is measured already; `b` is not, so its size comes from a /measure.
        ds = await _ds(db, imgs, captions={a: "on the beach"}, scores={a: 0.9},
                       faces={a: {"width": 1024, "height": 768, "face_px": 300.0}})
        svc = svc or _UpService()
        s3 = _S3()
        _wire(monkeypatch, svc, s3)
        dataset_upscale.RUNS.pop(ds.id, None)
        return ds, imgs, svc, s3

    async def test_it_adds_a_copy_beside_each_source(self, db, monkeypatch, shared_session):
        from app.routes import datasets as mod
        ds, (a, b, big), svc, s3 = await self._world(db, monkeypatch)
        await mod.upscale_images_job(ds.id, [a, b], 2)
        run = dataset_upscale.RUNS[ds.id]
        assert run["error"] is None and run["running"] is False and run["stage"] == "done"
        assert (run["done"], run["total"]) == (2, 2)
        await db.refresh(ds)
        ups = [u for u in s3.uploads if "/upscaled-x2-" in u]
        assert len(ups) == 2
        # Beside its source, and the source kept.
        assert ds.images == [a, ups[0], b, ups[1], big]
        # One image per call, at 2x the long edge.
        assert [p["target_edge"] for p in svc.upscale_bodies] == [2048, 2400]
        assert all(len(p["images"]) == 1 for p in svc.upscale_bodies)
        # The same photograph: its caption and score; measured; lineage "upscale".
        assert ds.captions[ups[0]] == "on the beach" and ds.scores[ups[0]] == 0.9
        assert ds.faces[ups[0]]["width"] == 819
        assert "upscaled_from" not in ds.faces[ups[0]]
        assert {u: (e["from"], e["how"]) for u, e in ds.derived.items()} == {
            ups[0]: (a, "upscale"), ups[1]: (b, "upscale")}
        assert "Upscaled 2 image(s) 2x" in ds.notes

    async def test_an_image_at_the_ceiling_is_declined_not_sent(
            self, db, monkeypatch, shared_session):
        from app.routes import datasets as mod
        ds, (a, b, big), svc, s3 = await self._world(db, monkeypatch, prefix="u2")
        await mod.upscale_images_job(ds.id, [big], 4)
        await db.refresh(ds)
        assert ds.images == [a, b, big] and not svc.upscale_bodies
        assert "1 was left as is" in dataset_upscale.RUNS[ds.id]["summary"]

    async def test_a_service_that_ignores_target_edge_stores_nothing(
            self, db, monkeypatch, shared_session):
        from app.routes import datasets as mod
        ds, (a, b, big), svc, s3 = await self._world(
            db, monkeypatch, svc=_UpService(ignores_target=True), prefix="u3")
        await mod.upscale_images_job(ds.id, [a], 2)
        await db.refresh(ds)
        assert ds.images == [a, b, big] and not s3.uploads
        assert "wanly-gpu-docker#214" in dataset_upscale.RUNS[ds.id]["error"]

    async def test_a_source_removed_meanwhile_gets_no_copy(
            self, db, monkeypatch, shared_session):
        from app.routes import datasets as mod
        ds, (a, b, big), svc, s3 = await self._world(db, monkeypatch, prefix="u4")
        real = dataset_upscale.run_upscale

        async def remove_a_meanwhile(*args, **kw):
            out = await real(*args, **kw)
            ds.images = [b, big]
            await db.commit()
            return out
        monkeypatch.setattr(dataset_upscale, "run_upscale", remove_a_meanwhile)
        await mod.upscale_images_job(ds.id, [a], 2)
        await db.refresh(ds)
        assert ds.images == [b, big]

    async def test_it_starts_in_the_background(self, db, monkeypatch):
        from fastapi import BackgroundTasks
        from app.routes import datasets as mod
        from app.schemas.datasets import DatasetUpscaleRequest
        ds, (a, b, big), svc, s3 = await self._world(db, monkeypatch, prefix="u5")
        tasks = BackgroundTasks()
        out = await mod.upscale_images(ds.id, DatasetUpscaleRequest(uris=[a, a], factor=4),
                                       tasks, _user=None, db=db)
        assert out.running is True and out.total == 1 and len(tasks.tasks) == 1
        again = BackgroundTasks()
        assert (await mod.upscale_images(ds.id, DatasetUpscaleRequest(uris=[a]), again,
                                         _user=None, db=db)).running
        assert not again.tasks
        dataset_upscale.RUNS.pop(ds.id, None)

    async def test_bad_requests_are_refused_up_front(self, db, monkeypatch):
        from datetime import datetime, timezone
        from fastapi import BackgroundTasks, HTTPException
        from app.routes import datasets as mod
        from app.schemas.datasets import DatasetUpscaleRequest
        ds, (a, b, big), svc, s3 = await self._world(db, monkeypatch, prefix="u6")
        call = lambda uris: mod.upscale_images(ds.id, DatasetUpscaleRequest(uris=uris),
                                               BackgroundTasks(), _user=None, db=db)
        with pytest.raises(HTTPException) as e:
            await call(["s3://wanly-images/elsewhere.jpg"])
        assert e.value.status_code == 422
        clip = "s3://wanly-images/datasets/u6/c.mp4"
        ds.images = [*ds.images, clip]
        await db.commit()
        with pytest.raises(HTTPException) as e:
            await call([clip])
        assert e.value.status_code == 422
        # An older service: refused before anything starts.
        _wire(monkeypatch, _UpService(no_target_feature=True), s3)
        with pytest.raises(HTTPException) as e:
            await call([a])
        assert e.value.status_code == 503 and "#214" in e.value.detail
        ds.locked_at = datetime.now(timezone.utc)
        await db.commit()
        with pytest.raises(HTTPException) as e:
            await call([a])
        assert e.value.status_code == 409
        assert ds.id not in dataset_upscale.RUNS
