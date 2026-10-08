"""POST /datasets/{id}/unlock lifts a hand lock (#358, #363).

Training no longer locks a set (#420), so the hand lock -- optional, off by default -- is the
only lock /unlock has to lift. It records `unlocked_at`; the runs that trained on the set stay
listed, because the runs are the record now (#419).
"""
import importlib.util
import uuid
from pathlib import Path

import pytest

from tests.test_dataset_lock import _ds, _patch, _run, _Usr, bucket  # noqa: F401
from tests.test_dataset_manual_lock import _lock


async def _unlock(db, ds):
    from app.routes.datasets import unlock_dataset
    return await unlock_dataset(ds.id, _user=None, db=db)


@pytest.mark.asyncio
class TestUnlock:
    async def test_it_clears_a_hand_lock(self, db):
        from app.models import Dataset
        ds = await _ds(db)
        await _lock(db, ds, reason="reference")
        out = await _unlock(db, ds)
        assert out.locked is False and out.locked_at is None and out.locked_reason is None
        row = await db.get(Dataset, ds.id)
        assert row.locked_at is None and row.locked_reason is None and row.unlocked_at
        assert (await _patch(db, ds, images=ds.images[:2])).images == ds.images[:2]

    async def test_a_trained_set_needs_no_unlock(self, db):
        """#420: training locks nothing, so there is nothing for /unlock to lift."""
        from app.routes.datasets import get_dataset
        ds = await _ds(db)
        await _run(db, ds)
        out = await get_dataset(ds.id, db=db)
        assert out.locked is False and out.trained_by
        assert (await _patch(db, ds, images=ds.images[:2])).images == ds.images[:2]

    async def test_unlocking_keeps_the_runs_listed(self, db):
        """The runs are the record; an unlock no longer makes a set forget them."""
        ds = await _ds(db)
        job = await _run(db, ds)
        await _lock(db, ds, reason="v5 shipped")
        out = await _unlock(db, ds)
        assert out.locked is False and [t.job_id for t in out.trained_by] == [str(job.id)]

    async def test_locked_by_hand_after_the_unlock_stays_locked(self, db):
        from fastapi import HTTPException
        ds = await _ds(db)
        await _unlock(db, ds)
        await _lock(db, ds, reason="done tweaking")
        with pytest.raises(HTTPException) as e:
            await _patch(db, ds, images=ds.images[1:])
        assert e.value.detail == (
            f"{ds.name!r} is locked by hand (done tweaking) — unlock it to make changes")

    async def test_unlocking_again_moves_the_time(self, db):
        ds = await _ds(db)
        first = await _unlock(db, ds)
        second = await _unlock(db, ds)
        assert second.unlocked_at > first.unlocked_at and second.locked is False

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

    async def test_patch_cannot_set_it_or_lift_a_hand_lock(self, db):
        from app.models import Dataset
        from app.routes.datasets import update_dataset
        from app.schemas.datasets import DatasetUpdate
        ds = await _ds(db)
        await _lock(db, ds, reason="r")
        out = await update_dataset(
            ds.id, DatasetUpdate.model_validate({"unlocked_at": "2999-01-01T00:00:00Z",
                                                 "locked_at": None, "notes": "n"}),
            _user=None, db=db)
        assert out.locked is True and out.unlocked_at is None
        assert (await db.get(Dataset, ds.id)).unlocked_at is None


@pytest.mark.asyncio
async def test_over_http(db):
    from httpx import ASGITransport, AsyncClient
    from app.auth import get_current_user, verify_api_key_or_bearer
    from app.database import get_db
    from app.main import app
    ds = await _ds(db)
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
