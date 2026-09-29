"""A dataset can be locked by hand, trained or not; a clone is still the way out (#358).

#356 locks a set once a run has trained on it. Some sets need freezing without one, so
POST /datasets/{id}/lock sets `locked_at` / `locked_reason`, and from then on every #356
refusal applies. Only the one-time /unlock (#363, tests/test_dataset_unlock.py) lifts it,
a second lock changes nothing, and a clone starts unlocked.
"""
import importlib.util
import uuid
from pathlib import Path

import pytest

from app.enums import TrainingStatus
from tests.test_dataset_lock import _Upload, _Usr, _ds, _patch, _run, bucket  # noqa: F401


async def _lock(db, ds, reason=None):
    from app.routes.datasets import lock_dataset
    from app.schemas.datasets import DatasetLock
    return await lock_dataset(ds.id, DatasetLock(reason=reason), _user=None, db=db)


@pytest.mark.asyncio
class TestLock:
    async def test_it_locks_an_untrained_set(self, db):
        from app.routes.datasets import get_dataset
        ds = await _ds(db)
        out = await _lock(db, ds, reason="  reference set  ")
        assert out.locked is True and out.trained_by == []
        assert out.locked_at is not None and out.locked_reason == "reference set"
        again = await get_dataset(ds.id, db=db)
        assert (again.locked, again.locked_at, again.locked_reason) == (
            True, out.locked_at, "reference set")

    async def test_no_reason_and_a_blank_reason_are_both_none(self, db):
        from app.routes.datasets import lock_dataset
        a, b = await _ds(db), await _ds(db)
        assert (await lock_dataset(a.id, None, _user=None, db=db)).locked_reason is None
        assert (await _lock(db, b, reason="   ")).locked_reason is None

    async def test_locking_again_changes_nothing(self, db):
        """The first lock's time and reason stand; a double-click cannot overwrite them."""
        ds = await _ds(db)
        first = await _lock(db, ds, reason="first")
        second = await _lock(db, ds, reason="second")
        assert (second.locked_at, second.locked_reason) == (first.locked_at, "first")

    async def test_a_trained_set_can_also_be_locked_by_hand(self, db):
        """...which is what keeps it locked if the run is later failed or cancelled."""
        from app.routes.datasets import get_dataset
        ds = await _ds(db)
        job = await _run(db, ds, status=TrainingStatus.RUNNING)
        out = await _lock(db, ds, reason="keep")
        assert out.locked and out.trained_by and out.locked_reason == "keep"
        job.status = TrainingStatus.CANCELLED
        await db.commit()
        after = await get_dataset(ds.id, db=db)
        assert after.locked is True and after.trained_by == []

    async def test_a_missing_set_is_404(self, db):
        from fastapi import HTTPException
        from app.routes.datasets import lock_dataset
        with pytest.raises(HTTPException) as e:
            await lock_dataset(uuid.uuid4(), None, _user=None, db=db)
        assert e.value.status_code == 404

    def test_the_reason_is_capped(self):
        from pydantic import ValidationError
        from app.schemas.datasets import DatasetLock
        DatasetLock(reason="x" * 500)
        with pytest.raises(ValidationError):
            DatasetLock(reason="x" * 501)

    async def test_patch_cannot_set_or_clear_it(self, db):
        """Only /unlock clears it: the fields are not in DatasetUpdate, and unknown fields are
        ignored."""
        from httpx import ASGITransport, AsyncClient
        from app.auth import get_current_user
        from app.database import get_db
        from app.main import app
        from app.models import Dataset
        locked, free = await _ds(db), await _ds(db)
        await _lock(db, locked, reason="frozen")
        app.dependency_overrides[get_db] = lambda: db
        app.dependency_overrides[get_current_user] = lambda: _Usr()
        try:
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
                r1 = await c.patch(f"/datasets/{locked.id}",
                                   json={"locked_at": None, "locked_reason": None, "notes": "n"})
                r2 = await c.patch(f"/datasets/{free.id}",
                                   json={"locked_at": "2026-01-01T00:00:00Z",
                                         "locked_reason": "sneaky"})
        finally:
            app.dependency_overrides.clear()
        assert r1.status_code == 200 and r1.json()["locked"] is True
        assert r1.json()["locked_reason"] == "frozen"
        assert r2.status_code == 200 and r2.json()["locked"] is False
        assert r2.json()["locked_at"] is None
        await db.refresh(free)
        assert free.locked_at is None and free.locked_reason is None
        assert (await db.get(Dataset, locked.id)).locked_at is not None

    async def test_over_http(self, db):
        from httpx import ASGITransport, AsyncClient
        from app.auth import get_current_user, verify_api_key_or_bearer
        from app.database import get_db
        from app.main import app
        a, b = await _ds(db), await _ds(db)
        app.dependency_overrides[get_db] = lambda: db
        app.dependency_overrides[get_current_user] = lambda: _Usr()
        app.dependency_overrides[verify_api_key_or_bearer] = lambda: None
        try:
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
                r = await c.post(f"/datasets/{a.id}/lock", json={"reason": "reference"})
                bare = await c.post(f"/datasets/{b.id}/lock")
                long = await c.post(f"/datasets/{b.id}/lock", json={"reason": "x" * 501})
                lst = await c.get("/datasets")
        finally:
            app.dependency_overrides.clear()
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["locked"] is True and body["locked_reason"] == "reference"
        assert body["locked_at"] and body["trained_by"] == []
        assert bare.status_code == 200 and bare.json()["locked_reason"] is None
        assert long.status_code == 422
        row = next(d for d in lst.json() if d["id"] == str(a.id))
        assert row["locked"] is True and row["locked_reason"] == "reference"


@pytest.mark.asyncio
class TestRefusedWhileLockedByHand:
    """Every #356 refusal, with a detail that says 'locked by hand'."""

    async def _locked(self, db, reason="reference set", **kw):
        ds = await _ds(db, name=f"Payton {uuid.uuid4().hex[:4]}", **kw)
        await _lock(db, ds, reason=reason)
        return ds

    @staticmethod
    def _is_lock(e, ds, reason="reference set"):
        assert e.value.status_code == 409
        assert e.value.detail == (
            f"{ds.name!r} is locked by hand ({reason}) — clone it to make changes")

    async def test_the_detail_without_a_reason(self, db):
        from fastapi import HTTPException
        ds = await self._locked(db, reason=None)
        with pytest.raises(HTTPException) as e:
            await _patch(db, ds, images=ds.images[1:])
        assert e.value.detail == f"{ds.name!r} is locked by hand — clone it to make changes"

    async def test_the_detail_when_trained_as_well(self, db):
        from fastapi import HTTPException
        ds = await _ds(db)
        await _run(db, ds, version=5)
        await _lock(db, ds, reason="v5 shipped")
        with pytest.raises(HTTPException) as e:
            await _patch(db, ds, images=ds.images[1:])
        assert e.value.detail == (
            f"{ds.name!r} trained Kelly-2000 v5 and is locked, and was also locked by hand "
            f"(v5 shipped) — clone it to make changes")

    async def test_images_owner_kind_and_class(self, db):
        from fastapi import HTTPException
        ds = await self._locked(db, kind="regularization", reg_class="woman")
        for fields in ({"images": ds.images[1:]}, {"reg_class": "man"}, {"kind": None}):
            with pytest.raises(HTTPException) as e:
                await _patch(db, ds, **fields)
            self._is_lock(e, ds)

    async def test_uploading_images(self, db, bucket):
        from fastapi import HTTPException
        from app.routes.datasets import add_images
        ds = await self._locked(db)
        with pytest.raises(HTTPException) as e:
            await add_images(ds.id, files=[_Upload()], _user=None, db=db)
        self._is_lock(e, ds)
        assert bucket.objects == {}

    async def test_cropping(self, db):
        from fastapi import HTTPException
        from app.routes.datasets import crop_faces
        ds = await self._locked(db)
        with pytest.raises(HTTPException) as e:
            await crop_faces(ds.id, largest_only=False, uris=None, save_as=False,
                             _user=None, db=db)
        self._is_lock(e, ds)

    async def test_captions(self, db):
        from fastapi import BackgroundTasks, HTTPException
        from app.routes.datasets import caption_dataset, edit_dataset_caption
        from app.schemas.datasets import DatasetCaptionEdit
        ds = await self._locked(db)
        tasks = BackgroundTasks()
        with pytest.raises(HTTPException) as e:
            await caption_dataset(ds.id, tasks, body=None, _user=None, db=db)
        self._is_lock(e, ds)
        assert tasks.tasks == []
        with pytest.raises(HTTPException) as e:
            await edit_dataset_caption(ds.id, DatasetCaptionEdit(uri=ds.images[0], caption="x"),
                                       _user=None, db=db)
        self._is_lock(e, ds)

    async def test_regularizing(self, db):
        from fastapi import HTTPException
        from app.routes.datasets import regularize_dataset
        from app.schemas.datasets import DatasetRegularize
        ds = await self._locked(db, kind="regularization", reg_class="woman")
        with pytest.raises(HTTPException) as e:
            await regularize_dataset(ds.id, DatasetRegularize(count=3), user=_Usr(), db=db)
        self._is_lock(e, ds)

    async def test_deleting(self, db, bucket):
        from fastapi import HTTPException
        from app.models import Dataset
        from app.routes.datasets import delete_dataset
        ds = await self._locked(db)
        for purge in (False, True):
            with pytest.raises(HTTPException) as e:
                await delete_dataset(ds.id, purge=purge, _user=None, db=db)
            self._is_lock(e, ds)
        assert await db.get(Dataset, ds.id) is not None and bucket.deleted == []

    async def test_rename_notes_tags_and_anchor_stay_open(self, db):
        ds = await self._locked(db)
        name = f"renamed-{uuid.uuid4().hex[:6]}"
        out = await _patch(db, ds, name=name, notes="n", tags="t", anchor_uri=ds.images[1],
                           images=list(ds.images))
        assert (out.name, out.notes, out.tags, out.anchor_uri) == (name, "n", "t", ds.images[1])
        assert out.locked is True and out.locked_reason == "reference set"

    async def test_a_caption_run_stops_when_the_set_is_locked_under_it(self, db, monkeypatch):
        from app.routes import captions as cap_mod
        from app.routes import datasets as mod
        ds = await _ds(db)
        calls = []

        async def _caption(db_, image, style=None, instruction=None, interactive=True):
            calls.append(1)
            if len(calls) == 2:
                await _lock(db, ds, reason="mid-run")
            return f"caption {len(calls)}", instruction

        class _S3:
            def download_bytes(self, uri):
                return b"x"
        monkeypatch.setattr(cap_mod, "caption_image_bytes", _caption)
        monkeypatch.setattr(mod, "s3", _S3())
        n = await mod.caption_dataset_images(db, ds.id, overwrite=False)
        await db.refresh(ds)
        assert n == 1 and len(ds.captions) == 1
        assert "locked by hand (mid-run)" in mod._CAPTION_RUNS.pop(ds.id)["error"]

    async def test_a_locked_pool_does_not_collect_late_renders(self, db, bucket):
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
        await _lock(db, ds)

        held = await regularize_status(ds.id, _user=None, db=db)
        await db.refresh(ds)
        assert (held.collected_now, held.running, ds.images) == (0, 1, [])
        assert j.status == JobStatus.PROCESSING


@pytest.mark.asyncio
class TestCloneOfAHandLockedSet:
    async def test_starts_unlocked_and_editable(self, db):
        from app.models import Dataset
        from app.routes.datasets import clone_dataset
        from app.schemas.datasets import DatasetClone
        src = await _ds(db)
        await _lock(db, src, reason="frozen")
        out = await clone_dataset(src.id, DatasetClone(name=f"c-{uuid.uuid4().hex[:6]}"),
                                  user=_Usr(), db=db)
        assert out.locked is False and out.locked_at is None and out.locked_reason is None
        row = await db.get(Dataset, out.id)
        assert row.locked_at is None and row.locked_reason is None
        assert (await _patch(db, row, images=row.images[:2])).images == src.images[:2]
        await db.refresh(src)
        assert src.locked_at is not None and len(src.images) == 4


@pytest.mark.asyncio
async def test_the_list_is_still_two_queries(db):
    """The hand lock is a column on the row, so it costs the list nothing."""
    from sqlalchemy import event
    from app.routes.datasets import list_datasets
    by_hand, trained, free = await _ds(db), await _ds(db), await _ds(db)
    await _lock(db, by_hand, reason="r")
    await _run(db, trained)

    statements = []
    sync_engine = db.bind.sync_engine if hasattr(db.bind, "sync_engine") else db.bind
    listen = lambda *a: statements.append(a[2])  # noqa: E731
    event.listen(sync_engine, "before_cursor_execute", listen)
    try:
        rows = await list_datasets(db=db)
    finally:
        event.remove(sync_engine, "before_cursor_execute", listen)
    by_id = {r.id: r for r in rows}
    assert by_id[by_hand.id].locked and by_id[by_hand.id].locked_reason == "r"
    assert by_id[trained.id].locked and by_id[trained.id].locked_at is None
    assert not by_id[free.id].locked
    assert len(statements) == 2


@pytest.mark.asyncio
async def test_migration_104_up_and_down(db_engine):
    """Down drops both columns, up puts them back as nullable timestamptz / text. DDL is
    transactional in postgres, so the whole thing is rolled back afterwards."""
    from alembic.migration import MigrationContext
    from alembic.operations import Operations
    from sqlalchemy import inspect

    spec = importlib.util.spec_from_file_location(
        "m104", Path("alembic/versions/104_dataset_manual_lock.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert (m.revision, m.down_revision) == ("104", "103")

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
    assert "locked_at" not in gone and "locked_reason" not in gone
    assert back["locked_at"]["nullable"] and back["locked_reason"]["nullable"]
    assert back["locked_at"]["type"].timezone is True
    assert type(back["locked_reason"]["type"]).__name__ == "TEXT"
