"""End-to-end coverage for GET /segments/next — the only door into the system.

Written after this endpoint returned 500 in production for every claim against a re-rolled job
(#196) and stopped the 3090 dead. The bug was one missing predicate, and nothing caught it because
nothing here called the endpoint: the existing tests covered its auth, and its stale-reclaim WHERE
clause rebuilt separately in a test file. A 296-line function that every generation passes through
had no test that ran it.

What makes this endpoint different from the rest of the API is the failure mode. A 500 anywhere
else is a broken page. Here the daemon polls, gets a 500, and the queue stops — no work runs and
nothing raises an alarm except a human noticing the GPU is idle. So these tests are deliberately
about the shapes that make it throw or return the wrong row, not about response formatting.

They drive the real route over HTTP against a real database, because the failure was a database
result-cardinality error that no amount of mocking would have produced.
"""

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from httpx import ASGITransport, AsyncClient

from app.auth import get_current_user, verify_api_key
from app.database import get_db
from app.enums import JobStatus, SegmentStatus
from app.main import app
from app.models import Job, Segment, User, Worker

WORKER_ID = uuid.UUID("95d5dffe-881b-46a6-bd5f-8930b9a66b75")


async def _user(db) -> User:
    user = User(username=str(uuid.uuid4()), password_hash="x")
    db.add(user)
    await db.flush()
    return user


async def _job(db, user=None, *, seed=1000, priority=0, status=JobStatus.PENDING,
               starting_image="s3://b/start.png") -> Job:
    user = user or await _user(db)
    job = Job(
        user_id=user.id, name="j", width=480, height=832, fps=16, seed=seed,
        priority=priority, status=status, starting_image=starting_image,
    )
    db.add(job)
    await db.flush()
    return job


async def _segment(db, job, index=0, *, status=SegmentStatus.PENDING, discarded=False,
                   last_frame=None, seed=None, start_image=None, reprocess_type=None,
                   created_at=None) -> Segment:
    seg = Segment(
        job_id=job.id, index=index, prompt="p", status=status, discarded=discarded,
        last_frame_path=last_frame, seed=seed, start_image=start_image,
        reprocess_type=reprocess_type,
    )
    if created_at is not None:
        seg.created_at = created_at
    db.add(seg)
    await db.flush()
    return seg


async def _claim(db, kind="gpu"):
    """Call the endpoint exactly as the daemon does."""
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[verify_api_key] = lambda: None
    try:
        # Every real request gets a fresh session; these share one so writes roll back together.
        for obj in list(db.identity_map.values()):
            if isinstance(obj, (Job, Segment, Worker)):
                db.expire(obj)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.get(
                "/segments/next",
                params={"worker_id": str(WORKER_ID), "worker_name": "3090.zero", "kind": kind},
            )
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
class TestTheIncident:
    async def test_a_re_rolled_job_with_a_successor_can_still_be_claimed(self, db):
        """The production 500 (#196), from the outside.

        A re-roll archives a take and puts its replacement at the same index, so a job rolled six
        times has seven rows at index 0. Resolving "the previous segment" by index alone then
        raises MultipleResultsFound — on the claim path, which stops the queue rather than
        degrading.
        """
        job = await _job(db)
        for _ in range(6):
            await _segment(db, job, 0, status=SegmentStatus.COMPLETED, discarded=True,
                           last_frame="s3://b/thrown-away.png")
        await _segment(db, job, 0, status=SegmentStatus.COMPLETED, last_frame="s3://b/kept.png")
        await _segment(db, job, 1)

        resp = await _claim(db)

        assert resp.status_code == 200, resp.text
        assert resp.json() is not None

    async def test_the_successor_starts_from_the_live_take_not_an_archived_one(self, db):
        # Getting one row is not enough — it has to be the right one. Continuing from a frame
        # that was thrown away would generate silently wrong video.
        job = await _job(db)
        await _segment(db, job, 0, status=SegmentStatus.COMPLETED, discarded=True,
                       last_frame="s3://b/thrown-away.png")
        await _segment(db, job, 0, status=SegmentStatus.COMPLETED, last_frame="s3://b/kept.png")
        await _segment(db, job, 1)

        body = (await _claim(db)).json()

        assert body["start_image"] == "s3://b/kept.png"


@pytest.mark.asyncio
class TestWhatGetsClaimed:
    async def test_a_pending_segment_is_claimed_and_marked(self, db):
        job = await _job(db)
        seg = await _segment(db, job, 0)

        body = (await _claim(db)).json()

        assert body["id"] == str(seg.id)
        await db.refresh(seg)
        assert seg.status == SegmentStatus.CLAIMED
        assert seg.worker_id == WORKER_ID
        assert seg.worker_name == "3090.zero"
        assert seg.claimed_at is not None

    async def test_nothing_to_do_returns_no_segment(self, db):
        await _job(db)
        resp = await _claim(db)
        assert resp.status_code == 200
        assert resp.json() is None

    async def test_a_discarded_segment_is_never_claimed(self, db):
        # An archived take is evidence, not work. Claiming one would regenerate a clip the
        # operator has already thrown away.
        job = await _job(db)
        await _segment(db, job, 0, discarded=True)

        assert (await _claim(db)).json() is None

    async def test_an_already_claimed_segment_is_not_handed_out_twice(self, db):
        job = await _job(db)
        await _segment(db, job, 0, status=SegmentStatus.CLAIMED)

        assert (await _claim(db)).json() is None

    async def test_the_job_moves_to_processing(self, db):
        job = await _job(db, status=JobStatus.PENDING)
        await _segment(db, job, 0)

        await _claim(db)

        await db.refresh(job)
        assert job.status == JobStatus.PROCESSING

    async def test_lower_priority_number_goes_first(self, db):
        user = await _user(db)
        later = await _job(db, user, priority=5)
        first = await _job(db, user, priority=1)
        await _segment(db, later, 0)
        wanted = await _segment(db, first, 0)

        body = (await _claim(db)).json()

        assert body["id"] == str(wanted.id)


@pytest.mark.asyncio
class TestStartImageResolution:
    async def test_segment_zero_starts_from_the_job_image(self, db):
        job = await _job(db, starting_image="s3://b/job-start.png")
        await _segment(db, job, 0)

        assert (await _claim(db)).json()["start_image"] == "s3://b/job-start.png"

    async def test_a_later_segment_starts_from_the_previous_last_frame(self, db):
        job = await _job(db)
        await _segment(db, job, 0, status=SegmentStatus.COMPLETED, last_frame="s3://b/0-last.png")
        await _segment(db, job, 1)

        assert (await _claim(db)).json()["start_image"] == "s3://b/0-last.png"

    async def test_an_explicit_start_image_wins(self, db):
        job = await _job(db)
        await _segment(db, job, 0, start_image="s3://b/chosen.png")

        assert (await _claim(db)).json()["start_image"] == "s3://b/chosen.png"


@pytest.mark.asyncio
class TestSeed:
    async def test_it_derives_from_the_job_when_the_segment_has_none(self, db):
        # seg0 is exactly job.seed, so the whole job reproduces from that one number.
        job = await _job(db, seed=1000)
        await _segment(db, job, 0)

        assert (await _claim(db)).json()["seed"] == 1000

    async def test_a_later_segment_inherits_the_jobs_seed_unchanged(self, db):
        """The seed is LOCKED across a chain: segment 1 runs on the same seed as segment 0.

        This asserted `job.seed + index` until the LTX migration. That was WAN reasoning --
        WAN drifted, so decorrelating later segments spread the drift around. Under LTX a
        continuation is meant to look like the same shot continuing, and the seed is the
        single biggest determinant of how a take looks; expression especially is seed-driven
        far more than it is LoRA-driven. Varying it per index is exactly the thing that made
        a continuation look like a different woman.
        """
        job = await _job(db, seed=1000)
        await _segment(db, job, 0, status=SegmentStatus.COMPLETED, last_frame="s3://b/0.png")
        await _segment(db, job, 1)

        assert (await _claim(db)).json()["seed"] == 1000

    async def test_a_segments_own_seed_wins(self, db):
        # What a re-roll writes: same position, different seed.
        job = await _job(db, seed=1000)
        await _segment(db, job, 0, seed=150488800771430)

        assert (await _claim(db)).json()["seed"] == 150488800771430

    async def test_a_re_roll_changes_the_seed_and_the_chain_follows_the_new_one(self, db):
        """Re-roll and Next Segment want opposite things from a seed, and both are right.

        A re-roll asks for a DIFFERENT take, so it must change the seed. A continuation asks
        for the SAME shot carrying on, so it must keep it. Both are served by one field:
        re-roll sets a new `job.seed`, and everything live derives from that.

        The seed the discarded take ran on is stamped onto it so the archive can answer
        "which seed gave me that one" -- and that stamped seed must never leak into a
        continuation, because it produced a take that is no longer in the chain.
        """
        job = await _job(db, seed=1000)
        # what a re-roll leaves behind: the old take stamped and discarded, a new job seed,
        # and a fresh take deriving from it.
        await _segment(db, job, 0, seed=1000, status=SegmentStatus.COMPLETED,
                       discarded=True, last_frame="s3://b/old.png")
        job.seed = 2000
        await _segment(db, job, 0, status=SegmentStatus.COMPLETED, last_frame="s3://b/0.png")
        await _segment(db, job, 1)

        # 2000, not 1000: the continuation follows the take that survived.
        assert (await _claim(db)).json()["seed"] == 2000

    async def test_the_worker_gets_an_integer_not_a_string(self, db):
        # The console's schema sends seeds as strings for display precision. This payload feeds
        # the sampler, and it must not inherit that.
        job = await _job(db, seed=1000)
        await _segment(db, job, 0)

        assert isinstance((await _claim(db)).json()["seed"], int)


@pytest.mark.asyncio
class TestKindRouting:
    async def test_gpu_does_not_claim_cpu_reprocess_carriers(self, db):
        # The CPU track runs concurrently with generation; a GPU worker taking a hologram
        # carrier would idle the GPU on CPU work.
        job = await _job(db)
        await _segment(db, job, 1000, reprocess_type="ar_hologram")

        assert (await _claim(db, kind="gpu")).json() is None

    async def test_hologram_claims_only_carriers(self, db):
        job = await _job(db)
        await _segment(db, job, 0)
        carrier = await _segment(db, job, 1000, reprocess_type="ar_hologram")

        body = (await _claim(db, kind="hologram")).json()

        assert body is not None and body["id"] == str(carrier.id)

    async def test_gpu_ignores_a_paused_job(self, db):
        job = await _job(db, status=JobStatus.PAUSED)
        await _segment(db, job, 0)

        assert (await _claim(db, kind="gpu")).json() is None


async def _worker(db, *, heartbeat_ago_minutes=0) -> Worker:
    worker = Worker(
        friendly_name="other-gpu", hostname="h", ip_address="10.0.0.2",
        last_heartbeat=datetime.now(timezone.utc) - timedelta(minutes=heartbeat_ago_minutes),
    )
    db.add(worker)
    await db.flush()
    return worker


@pytest.mark.asyncio
class TestStaleReclaim:
    async def test_work_orphaned_by_a_dead_worker_comes_back(self, db):
        """A crashed worker's segment must not sit CLAIMED forever — it is the only thing
        standing between a lost machine and a job that never finishes."""
        job = await _job(db, status=JobStatus.PROCESSING)
        stale = await _segment(db, job, 0, status=SegmentStatus.CLAIMED)
        stale.claimed_at = datetime.now(timezone.utc) - timedelta(minutes=45)
        stale.worker_id = uuid.uuid4()
        await db.flush()

        body = (await _claim(db)).json()

        # Reclaimed and handed straight back out to the asking worker.
        assert body is not None and body["id"] == str(stale.id)
        await db.refresh(stale)
        assert stale.worker_id == WORKER_ID

    async def test_a_live_workers_long_render_is_not_stolen(self, db):
        """The 2026-08-15 incident: renders crossed 30 minutes and the age rule handed every
        in-flight segment to the other GPU's poll. Both GPUs then rendered the same segment,
        which from the outside looked like 'only one GPU ever works the queue'."""
        other = await _worker(db)  # heartbeating right now, mid-render
        other_id = other.id
        # A worker that is really rendering reports online-busy — the daemon sets it on
        # receiving the claim, before it starts — and its segment has progress written
        # within seconds. The fixture used to leave both at their defaults (idle, empty),
        # which is indistinguishable from a claim whose response was lost, and is the exact
        # state wanly-api#242 reclaims. Modelled faithfully here so this guard tests the
        # incident it was written for rather than passing by omission.
        other.status = "online-busy"
        job = await _job(db, status=JobStatus.PROCESSING)
        held = await _segment(db, job, 0, status=SegmentStatus.PROCESSING)
        held.claimed_at = datetime.now(timezone.utc) - timedelta(minutes=45)
        held.worker_id = other_id
        held.progress_log = "[4/6] ltx-engine: Processing"
        await db.flush()

        body = (await _claim(db)).json()

        assert body is None
        await db.refresh(held)
        assert held.worker_id == other_id
        assert held.status == SegmentStatus.PROCESSING

    async def test_a_claim_whose_response_was_lost_comes_back(self, db):
        """wanly-api#242. GET /segments/next mutates: the segment is marked and assigned
        before the answer is sent, so a transport failure on the way back leaves it owned by
        a worker that never learned of it.

        That worker stays healthy and heartbeating, so neither dead-worker rule fires. Before
        this, the segment sat PROCESSING forever and the queue stopped beside an idle GPU.
        """
        idle = await _worker(db)                     # alive, beating, and genuinely idle
        idle.status = "online-idle"
        job = await _job(db, status=JobStatus.PROCESSING)
        orphan = await _segment(db, job, 0, status=SegmentStatus.PROCESSING)
        orphan.claimed_at = datetime.now(timezone.utc) - timedelta(minutes=3)
        orphan.worker_id = idle.id
        orphan.progress_log = None                   # it never started: nothing was written
        await db.flush()

        body = (await _claim(db)).json()

        assert body is not None and body["id"] == str(orphan.id)
        await db.refresh(orphan)
        assert orphan.worker_id == WORKER_ID

    async def test_an_idle_worker_that_HAS_written_progress_keeps_its_claim(self, db):
        """Progress is proof the worker received the segment and is working on it.

        The daemon's status push can fail and the heartbeat only re-pushes on a CHANGE, so
        "idle" can be a lie. An empty progress log cannot be, which is why the rule needs
        both — and why a segment with progress is never taken, whatever the status says.
        """
        worker = await _worker(db)
        worker.status = "online-idle"                # status is wrong, but...
        worker_id = worker.id                        # captured: the commit expires the object
        job = await _job(db, status=JobStatus.PROCESSING)
        held = await _segment(db, job, 0, status=SegmentStatus.PROCESSING)
        held.claimed_at = datetime.now(timezone.utc) - timedelta(minutes=10)
        held.worker_id = worker.id
        held.progress_log = "[1/6] Downloading start image..."   # ...it is demonstrably working
        await db.flush()

        body = (await _claim(db)).json()

        assert body is None
        await db.refresh(held)
        assert held.worker_id == worker_id

    async def test_a_fresh_claim_is_not_reclaimed_before_the_grace_period(self, db):
        """A worker that has just claimed has not had time to report busy or write progress.

        Too short a window and the reclaim races the very worker it belongs to, which is how
        two workers end up on one segment.
        """
        worker = await _worker(db)
        worker.status = "online-idle"
        worker_id = worker.id                        # captured: the commit expires the object
        job = await _job(db, status=JobStatus.PROCESSING)
        fresh = await _segment(db, job, 0, status=SegmentStatus.PROCESSING)
        fresh.claimed_at = datetime.now(timezone.utc) - timedelta(seconds=20)
        fresh.worker_id = worker.id
        await db.flush()

        body = (await _claim(db)).json()

        assert body is None
        await db.refresh(fresh)
        assert fresh.worker_id == worker_id

    async def test_a_worker_that_stopped_heartbeating_loses_its_claim(self, db):
        # The heartbeat rule needs no 30-minute wait: ~10 missed beats is already proof of death.
        dead = await _worker(db, heartbeat_ago_minutes=10)
        job = await _job(db, status=JobStatus.PROCESSING)
        held = await _segment(db, job, 0, status=SegmentStatus.PROCESSING)
        held.claimed_at = datetime.now(timezone.utc) - timedelta(minutes=6)
        held.worker_id = dead.id
        await db.flush()

        body = (await _claim(db)).json()

        assert body is not None and body["id"] == str(held.id)
        await db.refresh(held)
        assert held.worker_id == WORKER_ID


@pytest.mark.asyncio
class TestRetryCannotResurrectAnArchivedTake:
    async def test_retrying_a_discarded_segment_is_refused(self, db):
        """Retry is how a discarded segment could become PENDING again.

        Claiming refuses to hand out a discarded segment, so without this the retry would look
        like it worked and then sit PENDING forever — the worst of both: no clip, no error.
        """
        owner = await _user(db)
        job = await _job(db, owner)
        seg = await _segment(db, job, 0, status=SegmentStatus.FAILED, discarded=True)

        app.dependency_overrides[get_db] = lambda: db
        app.dependency_overrides[get_current_user] = lambda: owner
        try:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(f"/segments/{seg.id}/retry")
        finally:
            app.dependency_overrides.clear()

        assert resp.status_code == 400
        assert "discarded" in resp.json()["detail"].lower()
        await db.refresh(seg)
        assert seg.status == SegmentStatus.FAILED
