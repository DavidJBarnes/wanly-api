"""The describe queue: every upload is described, one at a time, and the line outlives the
process (console#727)."""

from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

import pytest

from app import caption_tickets, describe_queue
from app.models import ImageMeta
from tests.test_enhance_prompt import _client

A, B = (f"s3://wanly-images/q/{n}.png" for n in "ab")
T0 = datetime(2026, 11, 4, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _wired(db, monkeypatch):
    @asynccontextmanager
    async def session():
        yield db
    monkeypatch.setattr(describe_queue, "async_session", session)
    monkeypatch.setattr(describe_queue, "RETRY_WAIT_S", 0)
    describe_queue._attempts.clear()
    describe_queue.running = None
    describe_queue.counts.update(waiting=0, failed=0)


class _Ticket:
    def __init__(self, status, **flags):
        self.status = status
        self.error = flags.pop("error", None)
        self.busy = flags.pop("busy", False)
        self.unreadable = flags.pop("unreadable", False)
        self.withdrawn = flags.pop("withdrawn", False)

    async def wait(self):
        return self


def _captioner(monkeypatch, db, *outcomes):
    """Each ticket takes the next outcome: words to save, or a failed _Ticket."""
    calls = []

    def request(path, half=caption_tickets.SCENE, **kw):
        calls.append((path, kw.get("origin")))
        out = outcomes[len(calls) - 1]
        if isinstance(out, _Ticket):
            return out, False

        async def save():
            meta = await db.get(ImageMeta, path)
            caption_tickets.apply_scene(meta, out, "i")
        return _Saving(save), False
    monkeypatch.setattr(caption_tickets, "request", request)
    return calls


class _Saving(_Ticket):
    def __init__(self, save):
        super().__init__(caption_tickets.DONE)
        self._save = save

    async def wait(self):
        await self._save()
        return self


async def _queued(db, *paths, **meta):
    for n, p in enumerate(paths):
        db.add(ImageMeta(path=p, describe_queued_at=T0 + timedelta(seconds=n), **meta))
    await db.flush()


async def _row(db, path):
    return await db.get(ImageMeta, path, populate_existing=True)


async def test_it_describes_one_image_per_turn_oldest_first(db, monkeypatch):
    calls = _captioner(monkeypatch, db, "first", "second")
    await _queued(db, B, A)
    assert await describe_queue.run_one() is True
    assert await describe_queue.run_one() is True
    assert await describe_queue.run_one() is False
    assert calls == [(B, "auto"), (A, "auto")]
    a, b = await _row(db, A), await _row(db, B)
    assert (b.scene_description, a.scene_description) == ("first", "second")
    assert a.describe_queued_at is None and b.describe_queued_at is None


async def test_an_image_described_while_it_waited_is_not_captioned_again(db, monkeypatch):
    calls = _captioner(monkeypatch, db)
    await _queued(db, A, scene_description="already there")
    await describe_queue.run_one()
    assert calls == []
    a = await _row(db, A)
    assert a.describe_queued_at is None and a.scene_description == "already there"


async def test_a_busy_captioner_is_waited_out_without_counting(db, monkeypatch):
    busy = _Ticket(caption_tickets.FAILED, busy=True, error="rendering")
    _captioner(monkeypatch, db, *([busy] * 5), "words")
    await _queued(db, A)
    for _ in range(5):
        await describe_queue.run_one()
        assert (await _row(db, A)).describe_queued_at is not None
    await describe_queue.run_one()
    a = await _row(db, A)
    assert a.scene_description == "words" and a.describe_error is None


async def test_other_failures_get_three_tries_then_the_queue_moves_on(db, monkeypatch):
    boom = _Ticket(caption_tickets.FAILED, error="boom")
    _captioner(monkeypatch, db, boom, boom, boom, "b words")
    await _queued(db, A, B)
    await describe_queue.run_one()
    await describe_queue.run_one()
    assert (await _row(db, A)).describe_queued_at is not None, "should still be in line"
    await describe_queue.run_one()
    a = await _row(db, A)
    assert a.describe_queued_at is None and a.describe_error == "boom"
    await describe_queue.run_one()
    assert (await _row(db, B)).scene_description == "b words"
    assert describe_queue._attempts == {}


async def test_an_unreadable_image_fails_at_once(db, monkeypatch):
    _captioner(monkeypatch, db, _Ticket(caption_tickets.FAILED, unreadable=True, error="gone"))
    await _queued(db, A)
    await describe_queue.run_one()
    a = await _row(db, A)
    assert a.describe_queued_at is None and a.describe_error == "gone"


async def test_upload_queues_an_image_and_nothing_else(db, monkeypatch):
    from app.routes import images
    monkeypatch.setattr(images, "upload_bytes",
                        lambda data, key, bucket: f"s3://{bucket}/{key}")
    from app.auth import verify_api_key_or_bearer
    client, app = await _client(db)
    app.dependency_overrides[verify_api_key_or_bearer] = lambda: None
    try:
        async with client as c:
            img = await c.post("/images/upload", params={"filename": "x.png"},
                               data={"folder": "up"}, files={"file": ("x.png", b"png")})
            txt = await c.post("/images/upload", params={"filename": "x.txt"},
                               data={"folder": "up"}, files={"file": ("x.txt", b"hi")})
    finally:
        app.dependency_overrides.clear()
    assert (await _row(db, img.json()["path"])).describe_queued_at is not None
    assert await _row(db, txt.json()["path"]) is None


async def test_an_uploaded_row_stays_out_of_the_tagged_view(db):
    """A queued row has no tags, so the untagged triage view still lists it."""
    await _queued(db, A)
    assert not ((await _row(db, A)).tags or "").strip()


async def test_retry_puts_the_failures_back_in_line(db):
    db.add(ImageMeta(path=A, describe_error="boom"))
    db.add(ImageMeta(path=B, describe_error="gone"))
    await db.flush()
    from app.auth import verify_api_key_or_bearer
    client, app = await _client(db)
    app.dependency_overrides[verify_api_key_or_bearer] = lambda: None
    try:
        async with client as c:
            before = (await c.get("/images/describe-queue")).json()
            r = (await c.post("/images/describe-queue/retry")).json()
            q = (await c.get("/images/caption-queue")).json()
    finally:
        app.dependency_overrides.clear()
    assert [f["path"] for f in before["failed"]] == [A, B]
    assert r["retried"] == 2 and r["status"]["waiting"] == 2 and r["status"]["failed"] == []
    assert q["backlog"] == 2 and q["backlog_failed"] == 0
