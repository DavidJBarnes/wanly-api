"""wanly-gpu-daemon#189: a busy worker never looks offline, and never claims a LoRA it lacks.

The daemon registers BEFORE its LoRA sync now and reports what it is doing as `phase`. Two
API halves:

  * the phase is stored and returned on the worker (the console shows it in place of the
    status) -- replaced on every registration, kept when a beat omits it;
  * `?pending_loras=` on the claim keeps segments naming a LoRA still being synced PENDING,
    while every other segment flows as before.

DB-backed: the gate is a WHERE clause, and a predicate that compiles is not one that matches.
"""
import uuid
from datetime import datetime, timezone

import pytest
from httpx import ASGITransport, AsyncClient

from app.auth import verify_api_key
from app.database import get_db
from app.enums import JobStatus, SegmentStatus
from app.main import app
from app.models import Job, Segment, User, Worker
from app.routes.segments import pending_lora_names
from app.schemas.workers import WorkerHeartbeat, WorkerRegister, WorkerResponse

pytestmark = pytest.mark.asyncio

SYNCING = {"phase": "syncing_loras", "detail": "Downloading Payton_v2_final (358 / 650 MB)",
           "file": "Payton_v2_final.safetensors", "done_bytes": 358, "total_bytes": 650,
           "eta_s": 700, "files_remaining": 1}


async def _call(db, method, url, **kw):
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[verify_api_key] = lambda: None
    try:
        for obj in list(db.identity_map.values()):
            if isinstance(obj, (Worker, Job, Segment)):
                db.expire(obj)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            return await c.request(method, url, **kw)
    finally:
        app.dependency_overrides.clear()


def _register_body(name, **extra):
    return {"friendly_name": name, "hostname": "h", "ip_address": "10.0.0.1",
            "comfyui_running": True, **extra}


# ------------------------------------------------------------------- the phase on the row


async def test_older_workers_are_unaffected():
    assert WorkerHeartbeat(comfyui_running=True).phase is None
    assert WorkerRegister(friendly_name="a", hostname="h", ip_address="i").phase is None
    assert "phase" in WorkerResponse.model_fields


async def test_registration_stores_the_phase_and_the_response_carries_it(db):
    name = f"box-{uuid.uuid4().hex[:8]}"
    r = await _call(db, "POST", "/workers", json=_register_body(name, phase=SYNCING))
    assert r.status_code == 201, r.text
    assert r.json()["phase"]["phase"] == "syncing_loras"
    assert r.json()["phase"]["file"] == "Payton_v2_final.safetensors"


async def test_a_new_registration_replaces_the_phase_even_with_none(db):
    """An older daemon taking over the row must not inherit the supervisor's 'switching'."""
    name = f"box-{uuid.uuid4().hex[:8]}"
    await _call(db, "POST", "/workers", json=_register_body(
        name, phase={"phase": "switching_mode", "detail": "Switching edit → render"}))
    r = await _call(db, "POST", "/workers", json=_register_body(name))
    assert r.json()["phase"] is None


async def test_a_heartbeat_updates_it_and_omitting_it_keeps_it(db):
    name = f"box-{uuid.uuid4().hex[:8]}"
    wid = (await _call(db, "POST", "/workers", json=_register_body(name))).json()["id"]

    r = await _call(db, "POST", f"/workers/{wid}/heartbeat",
                    json={"comfyui_running": True, "phase": SYNCING})
    assert r.json()["phase"]["done_bytes"] == 358

    r = await _call(db, "POST", f"/workers/{wid}/heartbeat", json={"comfyui_running": True})
    assert r.json()["phase"]["done_bytes"] == 358, "a beat without a phase must not erase it"

    r = await _call(db, "POST", f"/workers/{wid}/heartbeat",
                    json={"comfyui_running": True, "phase": {"phase": "ready", "detail": "Ready"}})
    assert r.json()["phase"]["phase"] == "ready"


# ------------------------------------------------------------------- the claim gate


def test_both_spellings_are_matched():
    assert pending_lora_names("Payton_v2_final.safetensors, Kerry_v2") == [
        "Kerry_v2", "Kerry_v2.safetensors", "Payton_v2_final", "Payton_v2_final.safetensors"]
    assert pending_lora_names(None) == [] and pending_lora_names(" , none") == []


async def _fixture(db, recipes):
    user = User(username=str(uuid.uuid4()), password_hash="x")
    db.add(user)
    await db.flush()
    segs = []
    for recipe in recipes:
        job = Job(user_id=user.id, name="j", width=480, height=832, fps=16, seed=1,
                  priority=0, status=JobStatus.PENDING, starting_image="s3://b/s.png")
        db.add(job)
        await db.flush()
        seg = Segment(job_id=job.id, index=0, prompt="p", status=SegmentStatus.PENDING,
                      discarded=False, ltx_recipe=recipe)
        db.add(seg)
        await db.flush()
        segs.append(seg)
    worker = Worker(friendly_name=f"w-{uuid.uuid4().hex[:8]}", hostname="h",
                    ip_address="10.0.0.1", status="online-idle",
                    last_heartbeat=datetime.now(timezone.utc))
    db.add(worker)
    await db.flush()
    return segs, worker


async def _claim(db, worker, pending=None):
    params = {"worker_id": str(worker.id), "worker_name": "w", "kind": "gpu"}
    if pending is not None:
        params["pending_loras"] = pending
    return await _call(db, "GET", "/segments/next", params=params)


async def test_a_segment_naming_a_lora_still_syncing_stays_pending(db):
    """The incident: Payton_v2_final 358 MB into its download. Not claimed, not failed."""
    (seg,), worker = await _fixture(db, [
        {"characters": [{"name": "Payton", "char_lora": "Payton_v2_final"}]}])
    r = await _claim(db, worker, "Payton_v2_final.safetensors")
    assert r.status_code == 200, r.text
    assert r.json() is None
    await db.refresh(seg)
    assert seg.status == SegmentStatus.PENDING


async def test_other_work_flows_past_it(db):
    """Head-of-line: the held segment is older; the queue must not stop at it."""
    (held, other), worker = await _fixture(db, [
        {"char_lora": "Payton_v2_final"},
        {"characters": [{"name": "Karoline", "char_lora": "Karoline_v3_final"}],
         "content_loras": [{"name": "motion_v1"}]},
    ])
    r = await _claim(db, worker, "Payton_v2_final.safetensors")
    assert r.json()["id"] == str(other.id)


async def test_a_content_lora_still_syncing_holds_too(db):
    (seg,), worker = await _fixture(db, [{"content_loras": [{"name": "motion_v1"}]}])
    assert (await _claim(db, worker, "motion_v1.safetensors")).json() is None


async def test_a_prefix_is_not_a_match(db):
    """Exact string values only: syncing Payton_v2 must not hold Payton_v2_final."""
    (seg,), worker = await _fixture(db, [{"char_lora": "Payton_v2_final"}])
    assert (await _claim(db, worker, "Payton_v2")).json()["id"] == str(seg.id)


async def test_a_segment_without_a_recipe_is_never_held(db):
    """jsonb_path_exists(NULL) is NULL; NOT NULL would have hidden every WAN/free-form one."""
    (seg,), worker = await _fixture(db, [None])
    assert (await _claim(db, worker, "Payton_v2_final")).json()["id"] == str(seg.id)


async def test_the_lora_landing_releases_it(db):
    (seg,), worker = await _fixture(db, [{"char_lora": "Payton_v2_final"}])
    assert (await _claim(db, worker, "Payton_v2_final.safetensors")).json() is None
    assert (await _claim(db, worker, "")).json()["id"] == str(seg.id)
