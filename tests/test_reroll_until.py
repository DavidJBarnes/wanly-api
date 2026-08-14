"""'Re-roll until': the rule-driven re-roll loop (#342, console repo).

The loop is judged in the completion PATCH — the same request that carries the metrics — so
these tests drive it exactly the way the daemon does: create a take with a rule, PATCH it
completed with an identity_metrics blob, and look at what the API decided to do.

Run against a real database for the same reason as test_reroll_segment.py: what can go wrong
is database-shaped (a second live row at index 0, the cap setting read at judge time).
"""

import uuid

import pytest
from sqlalchemy import select

from app.enums import JobStatus, SegmentStatus
from app.main import app
from app.auth import get_current_user, verify_api_key
from app.database import get_db
from app.models import AppSetting, Job, Segment, User


# Eight points is the console's chip minimum, mirrored by the judge. Mean 3.0.
EXPRESSION_LOW = [3.0] * 8
# Mean 5.0 — clears a threshold of 4.
EXPRESSION_HIGH = [5.0] * 8


async def _user(db) -> User:
    user = User(username=str(uuid.uuid4()), password_hash="x")
    db.add(user)
    await db.flush()
    return user


async def _job_with_segment(db, user, **segment_kwargs) -> tuple[Job, Segment]:
    job = Job(
        user_id=user.id,
        name="job",
        width=480,
        height=832,
        fps=16,
        seed=1234,
        starting_image="s3://b/start.png",
        status=JobStatus.AWAITING,
    )
    db.add(job)
    await db.flush()

    fields = dict(
        job_id=job.id,
        index=0,
        prompt="she moves",
        duration_seconds=5.0,
        speed=1.0,
        status=SegmentStatus.COMPLETED,
        output_path="s3://b/out.mp4",
    )
    fields.update(segment_kwargs)
    segment = Segment(**fields)
    db.add(segment)
    await db.flush()
    return job, segment


def _expire(db):
    # Shared session across "requests" — expire so routes reload current rows. See
    # test_reroll_segment.py for why the user object is left alone.
    for obj in list(db.identity_map.values()):
        if isinstance(obj, (Job, Segment)):
            db.expire(obj)


async def _request(db, user, method, url, json=None):
    from httpx import ASGITransport, AsyncClient

    _expire(db)
    app.dependency_overrides[get_current_user] = lambda: user
    app.dependency_overrides[verify_api_key] = lambda: None
    app.dependency_overrides[get_db] = lambda: db
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.request(method, url, json=json)
    finally:
        app.dependency_overrides.clear()


async def _complete_with_metrics(db, user, segment_id, series):
    return await _request(
        db, user, "PATCH", f"/segments/{segment_id}",
        json={"status": "completed", "identity_metrics": {"series_expression": series}},
    )


async def _live_segments(db, job_id):
    _expire(db)
    rows = (await db.execute(select(Segment).where(Segment.job_id == job_id))).scalars().all()
    return [s for s in rows if not s.discarded]


@pytest.mark.asyncio
class TestRerollRuleRequest:
    async def test_a_rule_rides_the_new_take_with_count_1(self, db):
        user = await _user(db)
        job, _ = await _job_with_segment(db, user)

        resp = await _request(
            db, user, "POST", f"/jobs/{job.id}/reroll",
            json={"rule_metric": "expression", "rule_threshold": 4},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["reroll_rule_metric"] == "expression"
        assert body["reroll_rule_threshold"] == 4
        assert body["reroll_count"] == 1

    async def test_a_plain_reroll_carries_no_rule(self, db):
        user = await _user(db)
        job, _ = await _job_with_segment(db, user)

        body = (await _request(db, user, "POST", f"/jobs/{job.id}/reroll")).json()
        assert body["reroll_rule_metric"] is None
        assert body["reroll_count"] is None

    async def test_an_unknown_metric_is_refused(self, db):
        user = await _user(db)
        job, _ = await _job_with_segment(db, user)

        resp = await _request(
            db, user, "POST", f"/jobs/{job.id}/reroll",
            json={"rule_metric": "vibes", "rule_threshold": 4},
        )
        assert resp.status_code == 400

    async def test_a_metric_without_a_threshold_is_refused(self, db):
        user = await _user(db)
        job, _ = await _job_with_segment(db, user)

        resp = await _request(
            db, user, "POST", f"/jobs/{job.id}/reroll",
            json={"rule_metric": "expression"},
        )
        assert resp.status_code == 400


@pytest.mark.asyncio
class TestRerollRuleJudging:
    async def _ruled_take(self, db, user, count=1, **kwargs):
        return await _job_with_segment(
            db, user,
            status=SegmentStatus.PROCESSING,
            reroll_rule_metric="expression",
            reroll_rule_threshold=4.0,
            reroll_count=count,
            **kwargs,
        )

    async def test_a_miss_archives_the_take_and_rolls_the_next_one(self, db):
        user = await _user(db)
        job, take = await self._ruled_take(db, user)

        resp = await _complete_with_metrics(db, user, take.id, EXPRESSION_LOW)
        assert resp.status_code == 200, resp.text

        live = await _live_segments(db, job.id)
        assert len(live) == 1
        fresh = live[0]
        assert fresh.id != take.id and fresh.index == 0
        assert fresh.status == SegmentStatus.PENDING
        # The rule travels with the chain, and the count advances.
        assert fresh.reroll_rule_metric == "expression"
        assert fresh.reroll_rule_threshold == 4.0
        assert fresh.reroll_count == 2

        await db.refresh(job)
        assert job.status == JobStatus.PENDING
        await db.refresh(take)
        assert take.discarded is True
        # The archived take keeps the metrics that condemned it, and its seed (see #199).
        assert take.identity_metrics == {"series_expression": EXPRESSION_LOW}
        assert take.seed == 1234

    async def test_a_met_rule_ends_the_loop(self, db):
        user = await _user(db)
        job, take = await self._ruled_take(db, user)

        await _complete_with_metrics(db, user, take.id, EXPRESSION_HIGH)

        live = await _live_segments(db, job.id)
        assert [s.id for s in live] == [take.id]
        await db.refresh(job)
        assert job.status == JobStatus.AWAITING

    async def test_the_cap_ends_the_loop(self, db):
        user = await _user(db)
        job, take = await self._ruled_take(db, user, count=3)  # default cap

        await _complete_with_metrics(db, user, take.id, EXPRESSION_LOW)

        live = await _live_segments(db, job.id)
        assert [s.id for s in live] == [take.id]
        await db.refresh(job)
        assert job.status == JobStatus.AWAITING

    async def test_the_cap_is_read_from_settings_at_judge_time(self, db):
        user = await _user(db)
        db.add(AppSetting(key="max_rerolls_per_job", value="5"))
        await db.flush()
        job, take = await self._ruled_take(db, user, count=3)

        await _complete_with_metrics(db, user, take.id, EXPRESSION_LOW)

        live = await _live_segments(db, job.id)
        assert live[0].reroll_count == 4  # 3 < 5, so it rolled

    async def test_an_unevaluable_rule_sits_idle_instead_of_looping_blind(self, db):
        user = await _user(db)
        job, take = await self._ruled_take(db, user)

        # Series too short for the console to even draw a chip: nothing visible could gate.
        await _complete_with_metrics(db, user, take.id, [3.0] * 4)

        live = await _live_segments(db, job.id)
        assert [s.id for s in live] == [take.id]
        await db.refresh(job)
        assert job.status == JobStatus.AWAITING

    async def test_a_missing_series_sits_idle(self, db):
        user = await _user(db)
        job, take = await self._ruled_take(db, user)

        resp = await _request(
            db, user, "PATCH", f"/segments/{take.id}",
            json={"status": "completed", "identity_metrics": {"series": [0.9] * 8}},
        )
        assert resp.status_code == 200

        live = await _live_segments(db, job.id)
        assert [s.id for s in live] == [take.id]

    async def test_a_failed_generation_is_not_judged(self, db):
        user = await _user(db)
        job, take = await self._ruled_take(db, user)

        await _request(
            db, user, "PATCH", f"/segments/{take.id}",
            json={"status": "failed", "error_message": "boom"},
        )

        live = await _live_segments(db, job.id)
        assert [s.id for s in live] == [take.id]
        await db.refresh(job)
        assert job.status == JobStatus.FAILED

    async def test_a_successor_segment_blocks_the_automatic_roll(self, db):
        # Cannot happen through the UI today, but an automatic roll that orphaned segment 1
        # would silently discard the frame it was generated from — so the judge verifies.
        user = await _user(db)
        job, take = await self._ruled_take(db, user)
        db.add(Segment(
            job_id=job.id, index=1, prompt="and continues",
            duration_seconds=5.0, speed=1.0, status=SegmentStatus.COMPLETED,
        ))
        await db.flush()

        await _complete_with_metrics(db, user, take.id, EXPRESSION_LOW)

        live = await _live_segments(db, job.id)
        assert {s.index for s in live} == {0, 1}
        assert all(not s.discarded for s in live)
