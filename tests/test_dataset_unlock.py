"""The one-time dataset unlock: reuse is free, changes need a deliberate unlock (#363).

A set is locked once a run trains on it (#356) or once it is locked by hand (#358).
POST /datasets/{id}/unlock sets `unlocked_at` and clears the hand lock; from then on the
training lock only counts runs created AFTER `unlocked_at`, so the runs so far stop locking
it and the next one locks it again. Every place that decides "locked" -- the response, the
409s, the caption loop and the regularization collector -- applies the same rule.
"""
import importlib.util
import uuid
from pathlib import Path

import pytest

from app.enums import TrainingStatus
from tests.test_dataset_lock import _ds, _patch, _run, _Usr, bucket  # noqa: F401
from tests.test_dataset_manual_lock import _lock


async def _unlock(db, ds):
    from app.routes.datasets import unlock_dataset
    return await unlock_dataset(ds.id, _user=None, db=db)


@pytest.mark.asyncio
class TestUnlock:
    async def test_it_makes_a_trained_set_editable(self, db, bucket):
        from app.models import Dataset
        from app.routes.datasets import delete_dataset, edit_dataset_caption, get_dataset
        from app.schemas.datasets import DatasetCaptionEdit
        ds = await _ds(db)
        job = await _run(db, ds, status=TrainingStatus.COMPLETED, version=7)
        assert (await get_dataset(ds.id, db=db)).locked

        out = await _unlock(db, ds)
        assert out.locked is False and out.trained_by == [] and out.unlocked_at is not None
        # The run itself is untouched: only the set lets go of it.
        await db.refresh(job)
        assert job.status == TrainingStatus.COMPLETED and job.dataset_images == ds.images

        before = list(ds.images)
        edited = await _patch(db, ds, images=before[1:])
        assert edited.images == before[1:] and edited.locked is False
        cap = await edit_dataset_caption(ds.id, DatasetCaptionEdit(uri=before[1], caption="a"),
                                         _user=None, db=db)
        assert cap.captions == {before[1]: "a"}
        await delete_dataset(ds.id, purge=False, _user=None, db=db)
        assert await db.get(Dataset, ds.id) is None

    async def test_a_run_created_after_the_unlock_locks_it_again(self, db):
        from fastapi import HTTPException
        from app.routes.datasets import get_dataset
        ds = await _ds(db)
        await _run(db, ds, version=5)
        await _unlock(db, ds)
        new = await _run(db, ds, status=TrainingStatus.PENDING, version=6)

        out = await get_dataset(ds.id, db=db)
        assert out.locked is True
        # Only the run after the unlock -- v5 is forgotten by the set, not by the Training page.
        assert [(r.job_id, r.version) for r in out.trained_by] == [(str(new.id), 6)]
        with pytest.raises(HTTPException) as e:
            await _patch(db, ds, images=ds.images[1:])
        assert e.value.status_code == 409
        assert e.value.detail == (
            f"{ds.name!r} trained Kelly-2000 v6 and is locked — clone it to make changes")

    async def test_a_failed_run_after_the_unlock_does_not_lock_it(self, db):
        from app.routes.datasets import get_dataset
        ds = await _ds(db)
        await _run(db, ds)
        await _unlock(db, ds)
        await _run(db, ds, status=TrainingStatus.FAILED)
        assert (await get_dataset(ds.id, db=db)).locked is False

    async def test_it_clears_a_hand_lock(self, db):
        from app.models import Dataset
        ds = await _ds(db)
        await _lock(db, ds, reason="reference")
        out = await _unlock(db, ds)
        assert out.locked is False and out.locked_at is None and out.locked_reason is None
        row = await db.get(Dataset, ds.id)
        assert row.locked_at is None and row.locked_reason is None and row.unlocked_at
        assert (await _patch(db, ds, images=ds.images[:2])).images == ds.images[:2]

    async def test_it_clears_both_locks_at_once(self, db):
        ds = await _ds(db)
        await _run(db, ds)
        await _lock(db, ds, reason="v5 shipped")
        out = await _unlock(db, ds)
        assert out.locked is False and out.trained_by == [] and out.locked_at is None

    async def test_locked_by_hand_after_the_unlock_stays_locked(self, db):
        from fastapi import HTTPException
        from app.routes.datasets import get_dataset
        ds = await _ds(db)
        await _run(db, ds)
        await _unlock(db, ds)
        await _lock(db, ds, reason="done tweaking")
        out = await get_dataset(ds.id, db=db)
        assert out.locked is True and out.trained_by == [] and out.unlocked_at is not None
        with pytest.raises(HTTPException) as e:
            await _patch(db, ds, images=ds.images[1:])
        assert e.value.detail == (
            f"{ds.name!r} is locked by hand (done tweaking) — clone it to make changes")

    async def test_unlocking_again_moves_the_cutoff(self, db):
        from app.routes.datasets import get_dataset
        ds = await _ds(db)
        first = await _unlock(db, ds)
        await _run(db, ds)
        assert (await get_dataset(ds.id, db=db)).locked
        second = await _unlock(db, ds)
        assert second.unlocked_at > first.unlocked_at and second.locked is False

    async def test_an_unlocked_set_that_was_never_locked_is_fine(self, db):
        ds = await _ds(db)
        out = await _unlock(db, ds)
        assert out.locked is False and out.unlocked_at is not None

    async def test_a_missing_set_is_404(self, db):
        from fastapi import HTTPException
        from app.routes.datasets import unlock_dataset
        with pytest.raises(HTTPException) as e:
            await unlock_dataset(uuid.uuid4(), _user=None, db=db)
        assert e.value.status_code == 404

    async def test_a_clone_does_not_copy_it(self, db):
        from app.routes.datasets import clone_dataset
        from app.schemas.datasets import DatasetClone
        src = await _ds(db)
        await _unlock(db, src)
        out = await clone_dataset(src.id, DatasetClone(name=f"c-{uuid.uuid4().hex[:6]}"),
                                  user=_Usr(), db=db)
        assert out.unlocked_at is None

    async def test_patch_cannot_set_it(self, db):
        """Only /unlock moves it: a PATCH that sends it is ignored, and the lock stands."""
        from app.models import Dataset
        ds = await _ds(db)
        await _run(db, ds)
        from app.routes.datasets import update_dataset
        from app.schemas.datasets import DatasetUpdate
        out = await update_dataset(
            ds.id, DatasetUpdate.model_validate({"unlocked_at": "2999-01-01T00:00:00Z",
                                                 "notes": "n"}), _user=None, db=db)
        assert out.locked is True and out.unlocked_at is None
        assert (await db.get(Dataset, ds.id)).unlocked_at is None


@pytest.mark.asyncio
class TestTheBackgroundPathsUseTheSameRule:
    async def test_the_caption_loop_runs_on_an_unlocked_set(self, db, monkeypatch):
        from app.routes import captions as cap_mod
        from app.routes import datasets as mod
        ds = await _ds(db)
        await _run(db, ds)
        await _unlock(db, ds)
        calls = []

        async def _caption(db_, image, style=None, instruction=None, interactive=True):
            calls.append(1)
            return f"caption {len(calls)}", instruction

        class _S3:
            def download_bytes(self, uri):
                return b"x"

            def head_object(self, uri):
                return {"Key": uri}
        monkeypatch.setattr(cap_mod, "caption_image_bytes", _caption)
        monkeypatch.setattr(mod, "s3", _S3())
        n = await mod.caption_dataset_images(db, ds.id, overwrite=False)
        await db.refresh(ds)
        assert n == 4 and len(ds.captions) == 4
        assert "error" not in mod._CAPTION_RUNS.pop(ds.id, {})

    async def test_the_caption_loop_stops_at_a_run_created_after_the_unlock(self, db,
                                                                              monkeypatch):
        from app.routes import captions as cap_mod
        from app.routes import datasets as mod
        ds = await _ds(db)
        await _run(db, ds)
        await _unlock(db, ds)
        calls = []

        async def _caption(db_, image, style=None, instruction=None, interactive=True):
            calls.append(1)
            if len(calls) == 2:
                await _run(db, ds, status=TrainingStatus.PENDING, version=77)
            return f"caption {len(calls)}", instruction

        class _S3:
            def download_bytes(self, uri):
                return b"x"

            def head_object(self, uri):
                return {"Key": uri}
        monkeypatch.setattr(cap_mod, "caption_image_bytes", _caption)
        monkeypatch.setattr(mod, "s3", _S3())
        n = await mod.caption_dataset_images(db, ds.id, overwrite=False)
        await db.refresh(ds)
        assert n == 1 and len(ds.captions) == 1
        assert "trained Kelly-2000 v77 and is locked" in mod._CAPTION_RUNS.pop(ds.id)["error"]

    async def test_an_unlocked_pool_collects_its_late_renders(self, db, bucket):
        from app.enums import JobStatus, SegmentStatus
        from app.models import Job, Segment, User
        from app.regularization import reg_tag
        from app.routes.datasets import regularize_status
        user = User(username=f"u{uuid.uuid4().hex[:6]}", password_hash="x")
        db.add(user)
        await db.flush()
        ds = await _ds(db, images=[], kind="regularization", reg_class="man")
        j = Job(user_id=user.id, name="r", width=832, height=1216, fps=24, seed=1,
                tags=f"regularization, {reg_tag(ds.id)}", status=JobStatus.PROCESSING)
        db.add(j)
        await db.flush()
        frame = f"s3://wanly-jobs/{j.id}/last_frame.png"
        bucket.objects[("wanly-jobs", f"{j.id}/last_frame.png")] = b"png"
        db.add(Segment(job_id=j.id, index=0, prompt="p", status=SegmentStatus.COMPLETED,
                       last_frame_path=frame))
        await db.commit()
        await _run(db, ds)

        held = await regularize_status(ds.id, _user=None, db=db)
        assert (held.collected_now, held.running) == (0, 1)
        await _unlock(db, ds)
        got = await regularize_status(ds.id, _user=None, db=db)
        await db.refresh(ds)
        assert got.collected_now == 1 and len(ds.images) == 1


@pytest.mark.asyncio
async def test_the_list_applies_each_sets_unlock_in_two_queries(db):
    from sqlalchemy import event
    from app.routes.datasets import list_datasets
    unlocked, relocked, trained = await _ds(db), await _ds(db), await _ds(db)
    for d in (unlocked, relocked, trained):
        await _run(db, d)
    await _unlock(db, unlocked)
    await _unlock(db, relocked)
    after = await _run(db, relocked)

    statements = []
    sync_engine = db.bind.sync_engine if hasattr(db.bind, "sync_engine") else db.bind
    listen = lambda *a: statements.append(a[2])  # noqa: E731
    event.listen(sync_engine, "before_cursor_execute", listen)
    try:
        rows = await list_datasets(db=db)
    finally:
        event.remove(sync_engine, "before_cursor_execute", listen)
    by_id = {r.id: r for r in rows}
    assert not by_id[unlocked.id].locked and by_id[unlocked.id].trained_by == []
    assert by_id[unlocked.id].unlocked_at is not None
    assert by_id[relocked.id].locked
    assert [r.job_id for r in by_id[relocked.id].trained_by] == [str(after.id)]
    assert by_id[trained.id].locked and by_id[trained.id].unlocked_at is None
    assert len(statements) == 2


@pytest.mark.asyncio
async def test_over_http(db):
    from httpx import ASGITransport, AsyncClient
    from app.auth import get_current_user, verify_api_key_or_bearer
    from app.database import get_db
    from app.main import app
    ds = await _ds(db)
    await _run(db, ds)
    await _lock(db, ds, reason="r")
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_current_user] = lambda: _Usr()
    app.dependency_overrides[verify_api_key_or_bearer] = lambda: None
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            r = await c.post(f"/datasets/{ds.id}/unlock")
            missing = await c.post(f"/datasets/{uuid.uuid4()}/unlock")
            g = await c.get(f"/datasets/{ds.id}")
    finally:
        app.dependency_overrides.clear()
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["locked"] is False and body["trained_by"] == []
    assert body["locked_at"] is None and body["locked_reason"] is None
    assert body["unlocked_at"]
    assert missing.status_code == 404
    assert g.json()["unlocked_at"] == body["unlocked_at"] and g.json()["locked"] is False


@pytest.mark.asyncio
async def test_unlock_needs_a_user(db):
    """Same auth as /lock: an API key alone (the daemon's) cannot unlock a set."""
    from httpx import ASGITransport, AsyncClient
    from app.database import get_db
    from app.main import app
    ds = await _ds(db)
    app.dependency_overrides[get_db] = lambda: db
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            r = await c.post(f"/datasets/{ds.id}/unlock")
    finally:
        app.dependency_overrides.clear()
    assert r.status_code in (401, 403)


@pytest.mark.asyncio
async def test_migration_106_up_and_down(db_engine):
    """Down drops the column, up puts it back as a nullable timestamptz. DDL is
    transactional in postgres, so the whole thing is rolled back afterwards."""
    from alembic.migration import MigrationContext
    from alembic.operations import Operations
    from sqlalchemy import inspect

    spec = importlib.util.spec_from_file_location(
        "m106", Path("alembic/versions/106_dataset_unlock.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert (m.revision, m.down_revision) == ("106", "105")

    def _cols(conn):
        return {c["name"]: c for c in inspect(conn).get_columns("datasets")}

    def _run_both(conn):
        with Operations.context(MigrationContext.configure(conn)):
            m.downgrade()
            gone = _cols(conn)
            m.upgrade()
            back = _cols(conn)
        return gone, back

    async with db_engine.connect() as conn:
        trans = await conn.begin()
        try:
            gone, back = await conn.run_sync(_run_both)
        finally:
            await trans.rollback()
    assert "unlocked_at" not in gone
    assert back["unlocked_at"]["nullable"] and back["unlocked_at"]["type"].timezone is True
