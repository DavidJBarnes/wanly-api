"""Where a newly queued segment lands: top keeps its job's slot, bottom sends it back (#471)."""
import uuid

import pytest

from app import queue_position
from app.models import AppSetting, Job, User
from app.queue_position import place_queued_job
from app.regularization import REG_PRIORITY_BASE
from app.routes.app_settings import _to_response


async def _jobs(db, *priorities):
    user = User(username=str(uuid.uuid4()), password_hash="x")
    db.add(user)
    await db.flush()
    jobs = [Job(user_id=user.id, name=f"j{p}", priority=p, width=640, height=640, fps=16, seed=1)
            for p in priorities]
    db.add_all(jobs)
    await db.flush()
    return jobs


async def _set(db, value):
    db.add(AppSetting(key=queue_position.SETTING_KEY, value=value))
    await db.flush()


@pytest.mark.asyncio
class TestPlaceQueuedJob:
    async def test_top_is_the_default_and_moves_nothing(self, db):
        first, _, _ = await _jobs(db, 0, 1, 2)
        await place_queued_job(db, first)
        assert first.priority == 0

    async def test_bottom_sends_the_job_behind_every_other_job(self, db):
        await _set(db, "bottom")
        first, _, last = await _jobs(db, 0, 1, 2)
        await place_queued_job(db, first)
        assert first.priority == last.priority + 1

    async def test_bottom_lands_ahead_of_the_regularization_pool(self, db):
        # The pool sits past REG_PRIORITY_BASE so real work never waits behind it.
        await _set(db, "bottom")
        first, last, reg = await _jobs(db, 0, 1, REG_PRIORITY_BASE)
        await place_queued_job(db, first)
        assert first.priority == 2 < reg.priority

    async def test_a_job_already_last_stays_put(self, db):
        await _set(db, "bottom")
        _, last = await _jobs(db, 0, 1)
        await place_queued_job(db, last)
        assert last.priority == 1

    async def test_a_regularization_render_is_never_pulled_forward(self, db):
        await _set(db, "bottom")
        _, reg = await _jobs(db, 0, REG_PRIORITY_BASE + 3)
        await place_queued_job(db, reg)
        assert reg.priority == REG_PRIORITY_BASE + 3


class TestSettingResponse:
    def test_defaults_to_top(self):
        assert _to_response({"negative_prompt": ""}).segment_queue_position == "top"

    def test_an_unknown_stored_value_reads_as_top(self):
        resp = _to_response({"negative_prompt": "", queue_position.SETTING_KEY: "middle"})
        assert resp.segment_queue_position == "top"

    def test_bottom_round_trips(self):
        resp = _to_response({"negative_prompt": "", queue_position.SETTING_KEY: "bottom"})
        assert resp.segment_queue_position == "bottom"
