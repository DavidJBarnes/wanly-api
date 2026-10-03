"""POST /images/in-use: the bulk-delete pre-check (console#594).

Bulk delete used to fire one DELETE per image and count the 409s, so the person learned
"M still in use" and nothing else. The pre-check answers for the whole selection in one call,
with names and with whether a holder still *needs* the file -- the difference between a dead
dataset entry and a queued render that will fail at claim time.

Against a real Postgres: the dataset half of the gate is JSONB matched in Python over real
rows, and the names come from joins a fake session would only pretend to do.
"""
import uuid
from unittest.mock import patch

import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.enums import JobStatus, SegmentStatus, TrainingStatus
from app.models import Dataset, Job, Segment, TrainingJob, User

BUCKET = "wanly-images"


def _uri(name: str) -> str:
    return f"s3://{BUCKET}/inuse-{uuid.uuid4().hex[:8]}/{name}.png"


@pytest_asyncio.fixture
async def api(db):
    from app.auth import get_current_user
    from app.database import get_db
    from app.main import app

    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_current_user] = lambda: object()
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            yield c
    finally:
        app.dependency_overrides.pop(get_db, None)
        app.dependency_overrides.pop(get_current_user, None)


async def _user(db) -> User:
    user = User(username=f"u{uuid.uuid4().hex[:8]}", password_hash="x")
    db.add(user)
    await db.flush()
    return user


async def _job(db, user, *, status=JobStatus.PENDING, starting_image=None, name="j") -> Job:
    job = Job(user_id=user.id, name=name, width=832, height=1216, fps=24, seed=7,
              status=status, starting_image=starting_image)
    db.add(job)
    await db.flush()
    return job


async def _segment(db, job, *, status=SegmentStatus.PENDING, index=0, start_image=None):
    seg = Segment(job_id=job.id, index=index, prompt="p", status=status,
                  start_image=start_image)
    db.add(seg)
    await db.flush()
    return seg


async def _dataset(db, images, name=None) -> Dataset:
    ds = Dataset(name=name or f"ds-{uuid.uuid4().hex[:6]}", images=list(images),
                 prefix=f"datasets/{uuid.uuid4().hex[:6]}")
    db.add(ds)
    await db.flush()
    return ds


async def _check(api, paths):
    resp = await api.post("/images/in-use", json={"paths": paths})
    assert resp.status_code == 200, resp.text
    return resp.json()


class TestImagesInUse:
    async def test_free_paths_are_absent_and_counted(self, db, api):
        free_a, free_b = _uri("a"), _uri("b")

        body = await _check(api, [free_a, free_b, free_a])

        # Duplicates are folded: the count is of distinct images.
        assert body == {"checked": 2, "in_use_count": 0, "paths": {}}

    async def test_dataset_holder_is_named_and_not_needed(self, db, api):
        held, free = _uri("held"), _uri("free")
        ds = await _dataset(db, [held], name=f"Kelly faces {uuid.uuid4().hex[:4]}")
        await db.commit()

        body = await _check(api, [held, free])

        assert body["in_use_count"] == 1
        assert free not in body["paths"]
        entry = body["paths"][held]
        assert entry["dataset_ids"] == [str(ds.id)]
        assert entry["datasets"] == [{"id": str(ds.id), "name": ds.name}]
        # Membership alone breaks nothing that is going to run.
        assert entry["needed"] is False

    async def test_queued_job_and_held_segment_are_needed(self, db, api):
        start, face = _uri("start"), _uri("face")
        user = await _user(db)
        queued = await _job(db, user, status=JobStatus.PENDING, starting_image=start,
                            name="queued render")
        parent = await _job(db, user, status=JobStatus.AWAITING, name="parent")
        seg = await _segment(db, parent, status=SegmentStatus.AWAITING_CAPTION, index=2,
                             start_image=face)
        await db.commit()

        body = await _check(api, [start, face])

        s = body["paths"][start]
        assert s["jobs"] == [{"id": str(queued.id), "name": "queued render",
                              "status": "pending", "state": "queued"}]
        assert s["needed"] is True

        f = body["paths"][face]
        assert f["segments"] == [{"id": str(seg.id), "job_id": str(parent.id), "index": 2,
                                  "status": "awaiting_caption", "state": "held",
                                  "job_name": "parent"}]
        assert f["needed"] is True

    async def test_archived_job_is_named_but_idle(self, db, api):
        """Archived jobs still count as holders (a re-run fetches the image), but nothing is
        going to run on its own, so it is not the force warning."""
        start = _uri("old")
        user = await _user(db)
        await _job(db, user, status=JobStatus.ARCHIVED, starting_image=start)
        await db.commit()

        entry = (await _check(api, [start]))["paths"][start]

        assert entry["jobs"][0]["state"] == "idle"
        assert entry["needed"] is False

    async def test_live_training_run_is_needed(self, db, api):
        img = _uri("train")
        run = TrainingJob(character=f"c{uuid.uuid4().hex[:6]}", trigger="t", version=1,
                          dataset_images=[img], status=TrainingStatus.PENDING, config={})
        db.add(run)
        await db.commit()

        entry = (await _check(api, [img]))["paths"][img]

        assert entry["training_ids"] == [str(run.id)]
        assert entry["trainings"][0]["state"] == "queued"
        assert entry["needed"] is True

    async def test_read_only(self, db, api):
        """A pre-check that changed anything would be a delete by another name."""
        held = _uri("held")
        ds = await _dataset(db, [held])
        await db.commit()

        with patch("app.routes.images.delete_object") as deleter:
            await _check(api, [held])

        deleter.assert_not_called()
        await db.refresh(ds)
        assert ds.images == [held]

    async def test_empty_selection_is_rejected(self, db, api):
        resp = await api.post("/images/in-use", json={"paths": []})
        assert resp.status_code == 422


class TestSingleDelete409NamesTraining:
    async def test_training_holder_is_in_the_409(self, db, api):
        """A live training run was part of the gate but missing from the 409 body, so the
        console saw a refusal with no holders and called it an ordinary error."""
        img = _uri("train")
        run = TrainingJob(character=f"c{uuid.uuid4().hex[:6]}", trigger="t", version=1,
                          dataset_images=[img], status=TrainingStatus.RUNNING, config={})
        db.add(run)
        await db.commit()

        with patch("app.routes.images.delete_object") as deleter:
            resp = await api.delete("/images", params={"path": img})

        assert resp.status_code == 409
        assert resp.json()["detail"]["training_ids"] == [str(run.id)]
        deleter.assert_not_called()
