"""What code a worker is actually running (wanly-gpu-docker#72).

Two workers ran different code for 14 hours and nothing said so. It surfaced as a 422 on a
content LoRA that looked random: the pod's daemon was current, the 3090's had been cloned
before the fix existed. Diagnosing it took SSH into both boxes and reading git.

TWO FIELDS, not one, because there are two update channels that drift separately:

    daemon_commit   re-cloned from main by start.sh at every container boot
    image_ref       start.sh, the downloader and the engine; only changes on pull + recreate

The 3090 proved the distinction the same day: `docker restart` moved daemon_commit to current
and left image_ref 37 hours stale. One field could not have shown that.
"""
import uuid

import pytest
from httpx import ASGITransport, AsyncClient

from app.auth import verify_api_key
from app.database import get_db
from app.main import app
from app.models import Worker
from app.schemas.workers import WorkerHeartbeat, WorkerResponse

pytestmark = pytest.mark.asyncio


async def _worker(db, **kw):
    w = Worker(friendly_name=str(uuid.uuid4())[:12], hostname="h", ip_address="10.0.0.1",
               status="online-idle", comfyui_running=True, **kw)
    db.add(w)
    await db.flush()
    return w


async def _beat(db, worker_id, **body):
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[verify_api_key] = lambda: None
    try:
        for obj in list(db.identity_map.values()):
            if isinstance(obj, Worker):
                db.expire(obj)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            return await c.post(f"/workers/{worker_id}/heartbeat",
                                json={"comfyui_running": True, **body})
    finally:
        app.dependency_overrides.clear()


async def test_an_older_daemon_still_heartbeats():
    """Required fields would 422 every worker on the previous daemon out of the pool the
    moment this deployed."""
    hb = WorkerHeartbeat(comfyui_running=True)
    assert hb.daemon_commit is None and hb.image_ref is None


async def test_a_heartbeat_records_both(db):
    w = await _worker(db)
    wid = w.id
    r = await _beat(db, wid, daemon_commit="a044cef", image_ref="sha256:01bfaa8a85a3")
    assert r.status_code == 200
    await db.refresh(w)
    assert w.daemon_commit == "a044cef"
    assert w.image_ref == "sha256:01bfaa8a85a3"


async def test_omitting_them_does_not_erase_what_was_reported(db):
    """An older daemon omits the field on every beat. Assigning None would blank a good value
    seconds after a newer worker reported it — the same trap loras and checkpoints hit."""
    w = await _worker(db, daemon_commit="a044cef", image_ref="sha256:aaa")
    wid = w.id
    assert (await _beat(db, wid)).status_code == 200
    await db.refresh(w)
    assert w.daemon_commit == "a044cef"
    assert w.image_ref == "sha256:aaa"


async def test_the_response_actually_carries_them(db):
    """THE point of the change, and the exact mistake fetchable_kinds made: a stored value a
    response model omits is silently indistinguishable from an unreported one. A field that
    exists to END invisibility must not itself be invisible."""
    assert "daemon_commit" in WorkerResponse.model_fields
    assert "image_ref" in WorkerResponse.model_fields

    w = await _worker(db, daemon_commit="a044cef", image_ref="sha256:aaa")
    wid = w.id
    r = await _beat(db, wid, daemon_commit="a044cef")
    assert r.json()["daemon_commit"] == "a044cef"
    assert r.json()["image_ref"] == "sha256:aaa"


async def test_the_two_fields_move_independently(db):
    """`docker restart` re-clones the daemon and reuses the image. A worker must be able to
    report a new commit on an old image — that is precisely the state the 3090 was in."""
    w = await _worker(db, daemon_commit="6786f86", image_ref="sha256:old")
    wid = w.id
    await _beat(db, wid, daemon_commit="a044cef")   # image_ref omitted
    await db.refresh(w)
    assert w.daemon_commit == "a044cef"
    assert w.image_ref == "sha256:old", "the image must not move when only the daemon did"
