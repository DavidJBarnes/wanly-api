"""#359: the size a recipe clip renders at, and the estimate priced from it.

The rule is a copy of derive_size() in wanly-gpu-docker's engine/app.py, and the cases in
TestTheEngineRule are that repo's tests/test_render_cap.py, value for value. If the engine's rule
moves, those tests move there first and these must be made to match them.

The DB tests pin what the copy is for: a job whose start frame was upscaled past the cap is
estimated, listed and described at the size the GPU actually renders, while its stored
width/height stay the start frame's.
"""

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from app.auth import get_current_user
from app.config import settings
from app.database import get_db
from app.enums import JobStatus, SegmentStatus
from app.estimation import MIN_SAMPLES, get_estimation_rates
from app.main import app
from app.models import Job, Segment, User
from app.render_size import effective_render_size, render_size, renders_through_recipe

CAP = 1024 * 1024


class TestTheEngineRule:
    def test_the_default_cap_is_the_engines(self):
        assert settings.max_render_pixels == CAP

    @pytest.mark.parametrize("w,h,want", [
        (1824, 1248, (1216, 832)),      # the upscaled landscape frame that prompted #148
        (1248, 1824, (832, 1216)),      # and its portrait twin
        (2432, 1664, (1216, 832)),      # 2x upscale, same aspect
        (1856, 1280, (1216, 832)),      # the size in #359's report
        (1280, 1856, (832, 1216)),
    ])
    def test_an_upscaled_frame_is_brought_back_to_the_normal_size(self, w, h, want):
        assert effective_render_size(w, h) == want

    @pytest.mark.parametrize("w,h", [(1216, 832), (832, 1216), (1024, 1024), (960, 544), (512, 768)])
    def test_a_frame_at_or_under_the_cap_renders_exactly_as_before(self, w, h):
        assert effective_render_size(w, h) == ((w // 64) * 64, (h // 64) * 64)

    @pytest.mark.parametrize("w,h", [(1824, 1248), (4000, 3000), (3000, 4000), (5312, 2988), (1300, 1300)])
    def test_the_result_stays_inside_the_cap_on_the_64_grid_and_keeps_the_aspect(self, w, h):
        rw, rh = effective_render_size(w, h)
        assert rw * rh <= CAP
        assert rw % 64 == 0 and rh % 64 == 0
        assert abs(rw / rh - w / h) < 0.08

    def test_it_never_scales_up(self):
        assert effective_render_size(300, 200) == (256, 192)

    def test_the_cap_is_adjustable(self):
        assert effective_render_size(1824, 1248, max_pixels=1824 * 1248) == (1792, 1216)
        assert effective_render_size(1824, 1248, max_pixels=0) == (1792, 1216)   # 0 = no cap

    def test_the_default_follows_the_setting(self, monkeypatch):
        monkeypatch.setattr(settings, "max_render_pixels", 0)
        assert effective_render_size(1824, 1248) == (1792, 1216)


class TestOnlyTheRecipePathIsCapped:
    @pytest.mark.parametrize("blob", [None, {}, {"recipe": None}, {"recipe": ""},
                                      {"character": "Kelly"}])
    def test_anything_but_a_named_recipe_renders_at_the_size_it_was_asked_for(self, blob):
        assert not renders_through_recipe(blob)
        assert render_size(1856, 1280, renders_through_recipe(blob)) == (1856, 1280)

    def test_a_named_recipe_is_capped(self):
        blob = {"recipe": "Standing", "character": "Kelly"}
        assert renders_through_recipe(blob)
        assert render_size(1856, 1280, renders_through_recipe(blob)) == (1216, 832)


# ---------------------------------------------------------------------------------------------
# Against a database: the fit, the per-job estimate, the queue total and the response fields.
# ---------------------------------------------------------------------------------------------

RECIPE = {"recipe": "Standing", "character": "Kelly"}
FPS = 24


async def _user(db) -> User:
    user = User(username=str(uuid.uuid4()), password_hash="x")
    db.add(user)
    await db.flush()
    return user


async def _job(db, user, w, h, *, status=JobStatus.FINALIZED) -> Job:
    job = Job(user_id=user.id, name="j", width=w, height=h, fps=FPS, seed=1,
              starting_image="s3://b/start.png", status=status)
    db.add(job)
    await db.flush()
    return job


async def _ran(db, user, w, h, seconds, *, recipe=RECIPE, runs=MIN_SAMPLES):
    """`runs` completed 10 s segments at w x h, each taking `seconds`."""
    now = datetime.now(timezone.utc)
    for i in range(runs):
        job = await _job(db, user, w, h)
        db.add(Segment(job_id=job.id, index=0, prompt="p", status=SegmentStatus.COMPLETED,
                       duration_seconds=10.0, ltx_recipe=recipe,
                       claimed_at=now - timedelta(hours=1, seconds=seconds),
                       completed_at=now - timedelta(hours=1)))
    await db.flush()


async def _queued(db, user, w, h, *, recipe=RECIPE) -> Job:
    job = await _job(db, user, w, h, status=JobStatus.PENDING)
    db.add(Segment(job_id=job.id, index=0, prompt="p", status=SegmentStatus.PENDING,
                   duration_seconds=10.0, ltx_recipe=recipe))
    await db.flush()
    return job


async def _get(db, user, path):
    from httpx import ASGITransport, AsyncClient

    # Jobs and segments only: the route must read them fresh, but the user is read outside a
    # greenlet by the dependency override, and expiring it makes that a lazy load.
    for obj in list(db.identity_map.values()):
        if isinstance(obj, (Job, Segment)):
            db.expire(obj)
    app.dependency_overrides[get_current_user] = lambda: user
    app.dependency_overrides[get_db] = lambda: db
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(path)
            assert response.status_code == 200, response.text
            return response.json()
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
class TestTheEstimateUsesTheRenderSize:
    async def test_capped_runs_are_filed_under_the_size_they_rendered_at(self, db):
        user = await _user(db)
        await _ran(db, user, 1856, 1280, 600)

        rates = await get_estimation_rates(db, user.id)

        assert (1216, 832, FPS) in rates["shape_rates"]
        assert (1856, 1280, FPS) not in rates["shape_rates"]

    async def test_a_non_recipe_job_keeps_its_own_size(self, db):
        user = await _user(db)
        await _ran(db, user, 1856, 1280, 600, recipe=None)

        rates = await get_estimation_rates(db, user.id)

        assert (1856, 1280, FPS) in rates["shape_rates"]

    async def test_capped_and_native_runs_pool_into_one_median(self, db):
        user = await _user(db)
        await _ran(db, user, 1216, 832, 600, runs=2)
        await _ran(db, user, 1856, 1280, 700, runs=1)

        rates = await get_estimation_rates(db, user.id)

        assert rates["shape_rates"][(1216, 832, FPS)] == (pytest.approx(60.0), 3)

    async def test_an_upscaled_job_is_priced_like_the_clip_it_renders(self, db):
        """The #359 report: a 1856x1280 start frame renders at 1216x832, so it costs what a
        1216x832 clip costs, not what its start frame's pixel count suggests."""
        user = await _user(db)
        await _ran(db, user, 1216, 832, 600)      # 60 s per second of clip
        await _ran(db, user, 1856, 1280, 1800, recipe=None)  # a genuinely big shape, 3x
        job = await _queued(db, user, 1856, 1280)

        body = await _get(db, user, f"/jobs/{job.id}")

        assert body["estimated_run_time"] == pytest.approx(600.0)
        assert body["segments"][0]["estimated_run_time"] == pytest.approx(600.0)

    async def test_the_job_list_prices_it_the_same_way(self, db):
        user = await _user(db)
        await _ran(db, user, 1216, 832, 600)
        job = await _queued(db, user, 1856, 1280)

        body = await _get(db, user, "/jobs?status=pending")

        item = next(i for i in body["items"] if i["id"] == str(job.id))
        assert item["estimated_run_time"] == pytest.approx(600.0)

    async def test_the_queue_total_too(self, db):
        user = await _user(db)
        await _ran(db, user, 1216, 832, 600)
        await _queued(db, user, 1856, 1280)
        await _queued(db, user, 1216, 832)

        body = await _get(db, user, "/stats")

        assert body["total_queue_time"] == pytest.approx(1200.0)

    async def test_a_non_recipe_job_is_still_priced_at_its_own_size(self, db):
        user = await _user(db)
        await _ran(db, user, 1216, 832, 600)
        job = await _queued(db, user, 1856, 1280, recipe=None)

        body = await _get(db, user, f"/jobs/{job.id}")

        # No history at 1856x1280 and too few shapes for a pixel law: honestly unpriced,
        # rather than borrowing the 1216x832 rate for a clip that really is bigger.
        assert body["estimated_run_time"] is None


@pytest.mark.asyncio
class TestTheResponseSaysWhatRenders:
    async def test_detail_carries_both_sizes(self, db):
        user = await _user(db)
        job = await _queued(db, user, 1856, 1280)

        body = await _get(db, user, f"/jobs/{job.id}")

        assert (body["width"], body["height"]) == (1856, 1280)
        assert (body["render_width"], body["render_height"]) == (1216, 832)

    async def test_list_carries_both_sizes(self, db):
        user = await _user(db)
        job = await _queued(db, user, 1280, 1856)

        body = await _get(db, user, "/jobs?status=pending")

        item = next(i for i in body["items"] if i["id"] == str(job.id))
        assert (item["width"], item["height"]) == (1280, 1856)
        assert (item["render_width"], item["render_height"]) == (832, 1216)

    async def test_an_uncapped_job_renders_at_its_own_size(self, db):
        user = await _user(db)
        job = await _queued(db, user, 1216, 832)

        body = await _get(db, user, f"/jobs/{job.id}")

        assert (body["render_width"], body["render_height"]) == (1216, 832)

    async def test_a_non_recipe_job_is_never_capped(self, db):
        user = await _user(db)
        job = await _queued(db, user, 1856, 1280, recipe=None)

        detail = await _get(db, user, f"/jobs/{job.id}")
        listed = await _get(db, user, "/jobs?status=pending")

        assert (detail["render_width"], detail["render_height"]) == (1856, 1280)
        item = next(i for i in listed["items"] if i["id"] == str(job.id))
        assert (item["render_width"], item["render_height"]) == (1856, 1280)

    async def test_the_stored_size_is_not_touched(self, db):
        user = await _user(db)
        job = await _queued(db, user, 1856, 1280)

        await _get(db, user, f"/jobs/{job.id}")
        await db.refresh(job)

        assert (job.width, job.height) == (1856, 1280)
