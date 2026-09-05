"""A worker is only handed segments whose models it can load (console#422).

Driven over HTTP against a real database, like test_claim_endpoint.py and for the same
reason: the gate is a WHERE clause, and a predicate that compiles is not a predicate that
matches. The failure it guards against is a pod claiming a 10Eros pose it cannot load —
after the claim, so the segment fails and the job stalls beside a 3090 that has the file.

The failure THIS could introduce is worse, which is why half these tests are about work
still flowing: a gate that matches nothing turns the whole fleet silent, and a silent fleet
is indistinguishable from an empty queue.
"""
import uuid
from datetime import datetime, timezone

import pytest
from httpx import ASGITransport, AsyncClient

from app.auth import get_current_user, verify_api_key
from app.database import get_db
from app.enums import JobStatus, SegmentStatus
from app.main import app
from app.models import Job, Segment, User, Worker

WORKER_ID = uuid.UUID("2f5a3c4e-0b1d-4a7e-9c88-1c2f3a4b5c6d")


async def _user(db) -> User:
    user = User(username=str(uuid.uuid4()), password_hash="x")
    db.add(user)
    await db.flush()
    return user


async def _job(db, user) -> Job:
    job = Job(user_id=user.id, name="j", width=480, height=832, fps=16, seed=1000,
              priority=0, status=JobStatus.PENDING, starting_image="s3://b/start.png")
    db.add(job)
    await db.flush()
    return job


async def _segment(db, job, *, recipe, index=0) -> Segment:
    seg = Segment(job_id=job.id, index=index, prompt="p", status=SegmentStatus.PENDING,
                  discarded=False, ltx_recipe=recipe)
    db.add(seg)
    await db.flush()
    return seg


async def _worker(db, *, checkpoints, fetchable=None) -> Worker:
    worker = Worker(
        id=WORKER_ID, friendly_name="pod-1", hostname="pod-1", ip_address="10.0.0.9",
        status="online-idle",
        last_heartbeat=datetime.now(timezone.utc), checkpoints=checkpoints,
        fetchable_kinds=fetchable,
    )
    db.add(worker)
    await db.flush()
    return worker


async def _claim(db):
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[verify_api_key] = lambda: None
    try:
        for obj in list(db.identity_map.values()):
            if isinstance(obj, (Job, Segment, Worker)):
                db.expire(obj)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            return await c.get("/segments/next", params={
                "worker_id": str(WORKER_ID), "worker_name": "pod-1", "kind": "gpu",
            })
    finally:
        app.dependency_overrides.clear()


async def _job_detail(db, owner, job):
    # Read the id BEFORE expiring: touching an expired attribute outside the route's own
    # greenlet triggers a synchronous lazy load and raises MissingGreenlet.
    url = f"/jobs/{job.id}"
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_current_user] = lambda: owner
    try:
        for obj in list(db.identity_map.values()):
            if isinstance(obj, (Job, Segment, Worker)):
                db.expire(obj)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            return await c.get(url)
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
class TestTheGate:
    async def test_a_pod_is_not_handed_a_checkpoint_it_lacks(self, db):
        """The incident: a pod holding only sulphur claimed a 10Eros pose, uploaded the start
        image, and ComfyUI rejected the graph across three loaders."""
        job = await _job(db, await _user(db))
        await _segment(db, job, recipe={"checkpoint": "10Eros_v1.5_bf16"})
        await _worker(db, checkpoints=["sulphur_dev_bf16"], fetchable=["lora"])

        resp = await _claim(db)

        assert resp.status_code == 200, resp.text
        assert resp.json() is None

    async def test_the_worker_that_has_it_still_gets_it(self, db):
        job = await _job(db, await _user(db))
        seg = await _segment(db, job, recipe={"checkpoint": "10Eros_v1.5_bf16"})
        await _worker(db, checkpoints=["sulphur_dev_bf16", "10Eros_v1.5_bf16"],
                      fetchable=["lora"])

        assert (await _claim(db)).json()["id"] == str(seg.id)

    async def test_a_blocked_segment_does_not_hide_a_runnable_one_behind_it(self, db):
        """Head-of-line blocking, which is why the gate is a WHERE clause and not a check on
        the row the query already picked. The older segment is unrunnable here; the queue
        must not stop at it."""
        user = await _user(db)
        blocked_job = await _job(db, user)
        await _segment(db, blocked_job, recipe={"checkpoint": "10Eros_v1.5_bf16"})
        ok_job = await _job(db, user)
        runnable = await _segment(db, ok_job, recipe={"checkpoint": "sulphur_dev_bf16"})
        await _worker(db, checkpoints=["sulphur_dev_bf16"], fetchable=["lora"])

        assert (await _claim(db)).json()["id"] == str(runnable.id)

    async def test_a_lora_it_has_never_seen_does_not_block_the_claim(self, db):
        """The daemon downloads those at claim time. Gating on one would refuse work the
        worker would have handled by itself."""
        job = await _job(db, await _user(db))
        seg = await _segment(db, job, recipe={"checkpoint": "sulphur_dev_bf16",
                                              "char_lora": "never_uploaded_v9"})
        await _worker(db, checkpoints=["sulphur_dev_bf16"], fetchable=["lora"])

        assert (await _claim(db)).json()["id"] == str(seg.id)

    async def test_a_worker_that_can_fetch_checkpoints_is_not_gated(self, db):
        """console#423's seam, from the outside."""
        job = await _job(db, await _user(db))
        seg = await _segment(db, job, recipe={"checkpoint": "10Eros_v1.5_bf16"})
        await _worker(db, checkpoints=["sulphur_dev_bf16"], fetchable=["lora", "checkpoint"])

        assert (await _claim(db)).json()["id"] == str(seg.id)


@pytest.mark.asyncio
class TestWorkStillFlows:
    async def test_an_older_daemon_reporting_no_inventory_still_claims(self, db):
        """NULL checkpoints means never reported, not none available. Starving it would look
        exactly like a dead queue on upgrade day."""
        job = await _job(db, await _user(db))
        seg = await _segment(db, job, recipe={"checkpoint": "10Eros_v1.5_bf16"})
        await _worker(db, checkpoints=None)

        assert (await _claim(db)).json()["id"] == str(seg.id)

    async def test_a_segment_with_no_recipe_is_never_gated(self, db):
        """A WAN segment or a free-form render declares no models, so it requires none."""
        job = await _job(db, await _user(db))
        seg = await _segment(db, job, recipe=None)
        await _worker(db, checkpoints=["sulphur_dev_bf16"], fetchable=["lora"])

        assert (await _claim(db)).json()["id"] == str(seg.id)

    async def test_a_recipe_without_a_checkpoint_runs_on_the_stack_default(self, db):
        """Segments predating per-pose base models must keep flowing."""
        job = await _job(db, await _user(db))
        seg = await _segment(db, job, recipe={"recipe": "old", "char_lora": "k3llydw_v2"})
        await _worker(db, checkpoints=["sulphur_dev_bf16"], fetchable=["lora"])

        assert (await _claim(db)).json()["id"] == str(seg.id)

    async def test_either_spelling_of_the_checkpoint_matches(self, db):
        job = await _job(db, await _user(db))
        seg = await _segment(db, job, recipe={"checkpoint": "sulphur_dev_bf16.safetensors"})
        await _worker(db, checkpoints=["sulphur_dev_bf16"], fetchable=["lora"])

        assert (await _claim(db)).json()["id"] == str(seg.id)


@pytest.mark.asyncio
class TestTheStallIsVisible:
    async def test_a_job_page_says_why_nothing_is_picking_it_up(self, db):
        """Without this the gate trades a loud failure for a silent one: PENDING forever,
        beside an idle fleet, looking exactly like an empty queue."""
        owner = await _user(db)
        job = await _job(db, owner)
        await _segment(db, job, recipe={"checkpoint": "10Eros_v1.5_bf16"})
        await _worker(db, checkpoints=["sulphur_dev_bf16"], fetchable=["lora"])

        body = (await _job_detail(db, owner, job)).json()

        assert body["segments"][0]["blocked_reason"] == \
            "No online worker can load checkpoint '10Eros_v1.5_bf16'"

    async def test_a_runnable_segment_says_nothing(self, db):
        owner = await _user(db)
        job = await _job(db, owner)
        await _segment(db, job, recipe={"checkpoint": "sulphur_dev_bf16"})
        await _worker(db, checkpoints=["sulphur_dev_bf16"], fetchable=["lora"])

        body = (await _job_detail(db, owner, job)).json()

        assert body["segments"][0]["blocked_reason"] is None

    async def test_no_reported_inventory_means_no_diagnosis(self, db):
        """"No online worker can load X" would be a guess dressed as a diagnosis when nothing
        has said what it holds."""
        owner = await _user(db)
        job = await _job(db, owner)
        await _segment(db, job, recipe={"checkpoint": "10Eros_v1.5_bf16"})
        await _worker(db, checkpoints=None)

        body = (await _job_detail(db, owner, job)).json()

        assert body["segments"][0]["blocked_reason"] is None
