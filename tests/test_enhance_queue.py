"""The enhancer's queue: images go in, one runs at a time, and it outlives the process (console#719)."""

from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

import pytest

from app import enhance_queue, prompt_enhance
from app.config import settings
from app.models import ACTION_GROUP, ImageMeta, TitleTag
from tests.test_enhance_prompt import _client

A, B, C = (f"s3://wanly-images/q/{n}.png" for n in "abc")
QUEUE = "/images/enhance-prompt/queue"


@pytest.fixture(autouse=True)
def _wired(db, monkeypatch):
    @asynccontextmanager
    async def session():
        yield db
    monkeypatch.setattr(enhance_queue, "async_session", session)
    monkeypatch.setattr(enhance_queue.s3, "download_bytes", lambda uri: b"png")
    monkeypatch.setattr(enhance_queue, "RETRY_WAIT_S", 0)
    monkeypatch.setattr(settings, "prompt_enhance_url", "http://enhancer")
    enhance_queue._attempts.clear()
    enhance_queue.running = None


def _enhancer(monkeypatch, *outcomes):
    """Each call takes the next outcome: a prompt, or an exception to raise."""
    calls = []

    async def fake(image, text):
        calls.append(text)
        out = outcomes[len(calls) - 1]
        if isinstance(out, Exception):
            raise out
        return {"prompt": out, "ms": 1, "cold": False}
    monkeypatch.setattr(prompt_enhance, "enhance", fake)
    return calls


async def _images(db, *paths, **meta):
    db.add(TitleTag(name="Missionary", group=ACTION_GROUP, description="He thrusts."))
    for p in paths:
        db.add(ImageMeta(path=p, tags="Missionary", **meta))
    await db.flush()


async def _call(db, method, **kw):
    client, app = await _client(db)
    try:
        async with client as c:
            return await c.request(method, QUEUE, **kw)
    finally:
        app.dependency_overrides.clear()


async def _add(db, *paths, redo=False):
    r = await _call(db, "POST", json={"paths": list(paths), "redo": redo})
    assert r.status_code == 200, r.text
    return r.json()


async def _row(db, path):
    return await db.get(ImageMeta, path, populate_existing=True)


async def test_adding_returns_at_once_and_keeps_the_order_asked(db, monkeypatch):
    calls = _enhancer(monkeypatch)
    await _images(db, A, B, C)
    body = await _add(db, C, A, B)
    assert body["queued"] == 3 and body["status"]["waiting"] == [C, A, B]
    assert calls == [], "adding must not wait on the enhancer"


async def test_it_runs_one_image_per_turn_oldest_first(db, monkeypatch):
    _enhancer(monkeypatch, "first", "second")
    await _images(db, A, B)
    await _add(db, B, A)
    assert await enhance_queue.run_one() is True
    assert (await _row(db, B)).enhanced_prompt == "first"
    assert (await _row(db, A)).enhanced_prompt is None
    assert await enhance_queue.run_one() is True
    assert (await _row(db, A)).enhanced_prompt == "second"
    assert await enhance_queue.run_one() is False
    assert (await _call(db, "GET")).json()["waiting"] == []


async def test_a_bulk_ask_does_not_overwrite_a_kept_prompt(db, monkeypatch):
    await _images(db, A, enhanced_prompt="kept")
    body = await _add(db, A)
    assert body["queued"] == 0 and body["already_enhanced"] == 1


async def test_redo_replaces_it(db, monkeypatch):
    _enhancer(monkeypatch, "new")
    await _images(db, A, enhanced_prompt="old")
    assert (await _add(db, A, redo=True))["queued"] == 1
    await enhance_queue.run_one()
    assert (await _row(db, A)).enhanced_prompt == "new"


async def test_an_image_with_nothing_to_expand_is_counted_not_queued(db):
    db.add(ImageMeta(path=A, tags="Nobody"))
    await db.flush()
    body = await _add(db, A, B)
    assert body["queued"] == 0 and body["no_action"] == 2


async def test_an_image_in_line_keeps_its_place(db):
    await _images(db, A, B)
    await _add(db, A, B)
    before = (await _row(db, A)).enhance_queued_at
    assert (await _add(db, A))["queued"] == 0
    assert (await _row(db, A)).enhance_queued_at == before


async def test_a_queued_row_is_all_a_new_process_needs(db, monkeypatch):
    """The point of keeping it on the row: nothing in memory is required to carry on."""
    _enhancer(monkeypatch, "done")
    await _images(db, A, enhance_queued_at=datetime.now(timezone.utc) - timedelta(hours=1))
    assert enhance_queue.running is None
    assert await enhance_queue.run_one() is True
    assert (await _row(db, A)).enhanced_prompt == "done"


async def test_an_outage_is_retried_then_reported_and_the_queue_moves_on(db, monkeypatch):
    down = prompt_enhance.EnhanceError("did not answer")
    _enhancer(monkeypatch, down, down, down, "b ok")
    await _images(db, A, B)
    await _add(db, A, B)
    await enhance_queue.run_one()
    await enhance_queue.run_one()
    assert (await _row(db, A)).enhance_queued_at is not None, "should still be in line"
    await enhance_queue.run_one()
    a = await _row(db, A)
    assert a.enhance_queued_at is None and a.enhance_error == "did not answer"
    await enhance_queue.run_one()
    assert (await _row(db, B)).enhanced_prompt == "b ok"
    status = (await _call(db, "GET")).json()
    assert status["failed"] == [{"path": A, "error": "did not answer"}]


async def test_a_recovery_clears_the_count(db, monkeypatch):
    down = prompt_enhance.EnhanceError("did not answer")
    _enhancer(monkeypatch, down, "ok")
    await _images(db, A)
    await _add(db, A)
    await enhance_queue.run_one()
    await enhance_queue.run_one()
    assert (await _row(db, A)).enhanced_prompt == "ok"
    assert enhance_queue._attempts == {}


async def test_an_unreadable_image_fails_without_retrying(db, monkeypatch):
    def gone(uri):
        raise OSError("no such key")
    monkeypatch.setattr(enhance_queue.s3, "download_bytes", gone)
    await _images(db, A)
    await _add(db, A)
    await enhance_queue.run_one()
    a = await _row(db, A)
    assert a.enhance_queued_at is None and "could not read" in a.enhance_error


async def test_asking_again_clears_a_failure(db):
    await _images(db, A, enhance_error="did not answer")
    await _add(db, A)
    a = await _row(db, A)
    assert a.enhance_error is None and a.enhance_queued_at is not None


async def test_cancel_empties_the_line_and_drops_a_result_in_flight(db, monkeypatch):
    await _images(db, A, B)
    await _add(db, A, B)

    async def cancelled_meanwhile(image, text):
        assert (await _call(db, "DELETE")).json() == {"running": None, "waiting": [], "failed": []}
        return {"prompt": "late", "ms": 1, "cold": False}
    monkeypatch.setattr(prompt_enhance, "enhance", cancelled_meanwhile)
    await enhance_queue.run_one()
    assert (await _row(db, A)).enhanced_prompt is None
    assert await enhance_queue.run_one() is False


async def test_switched_off_is_refused_up_front(db, monkeypatch):
    monkeypatch.setattr(settings, "prompt_enhance_url", "")
    await _images(db, A)
    assert (await _call(db, "POST", json={"paths": [A]})).status_code == 503
