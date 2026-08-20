"""Filtering videos (jobs) by whole tags, and the search box that no longer over-matches them.

Two issues, one fix. Videos could only be narrowed by a substring search over name AND tags
(console#352 asked for the image repo's tag pills; console#353 reported the fallout: searching
"AR" to find AR videos returned everything whose name or tags merely contained "ar").

So `tags` matches a tag in FULL and every pill ANDs, while `q` keeps matching the job NAME as a
fragment -- a half-remembered name is often the only handle there is -- but matches tags whole.

These run against real Postgres: the whole point is what the database does with commas and
NULLs, and a rendered-SQL assertion would test the string rather than the behaviour.
"""

import uuid

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

from app.auth import get_current_user
from app.database import get_db
from app.enums import JobStatus
from app.main import app
from app.models import Job, User
from app.routes.jobs import job_filter


@pytest.mark.asyncio
class TestJobFilter:
    async def _seed(self, db):
        user = User(username=str(uuid.uuid4()), password_hash="x")
        db.add(user)
        await db.flush()

        rows = [
            # name,                 tags,                status
            ("driveway ar tier1",   "AR, kelly",         JobStatus.FINALIZED),
            ("kitchen argentina",   "argentina, kelly",  JobStatus.FINALIZED),
            ("starlight bar scene", "starlight",         JobStatus.FINALIZED),
            ("kelly closeup",       "kelly",             JobStatus.FINALIZED),
            ("ar depth test",       "ar, depth",         JobStatus.PENDING),
            ("untagged take",       None,                JobStatus.FINALIZED),
        ]
        for name, tags, job_status in rows:
            db.add(Job(
                user_id=user.id, name=name, tags=tags, status=job_status,
                width=640, height=640, fps=60, seed=1000,
            ))
        await db.flush()
        return user

    async def _names(self, db, user, **kwargs):
        clauses = job_filter(user_id=user.id, **kwargs)
        rows = (await db.execute(select(Job.name).where(*clauses))).scalars().all()
        return set(rows)

    async def test_a_tag_pill_does_not_match_a_longer_tag_that_starts_with_it(self, db):
        # console#353: "AR" pulled in argentina and starlight.
        user = await self._seed(db)
        assert await self._names(db, user, status_filter="finalized", tags=["AR"]) == {
            "driveway ar tier1",
        }

    async def test_tag_pills_are_conjunctive(self, db):
        user = await self._seed(db)
        assert await self._names(db, user, status_filter="finalized,pending", tags=["ar", "depth"]) == {
            "ar depth test",
        }

    async def test_two_tags_no_job_carries_returns_nothing(self, db):
        # There is no OR: two pills mean "both tags on one job", and empty is the honest answer.
        user = await self._seed(db)
        assert await self._names(db, user, status_filter="finalized", tags=["AR", "starlight"]) == set()

    async def test_tag_matching_ignores_case_and_spacing(self, db):
        user = await self._seed(db)
        assert await self._names(db, user, status_filter="finalized", tags=[" Ar "]) == {
            "driveway ar tier1",
        }

    async def test_untagged_jobs_never_match_a_tag(self, db):
        # concat() is null-safe, so a NULL tags column fails to match rather than poisoning it.
        user = await self._seed(db)
        assert "untagged take" not in await self._names(db, user, status_filter="finalized", tags=["kelly"])

    async def test_search_matches_a_whole_tag_not_a_fragment_of_one(self, db):
        # The #353 fix, tag half: q=AR still finds the AR-*tagged* job, and no longer drags in
        # the one tagged `argentina` or `starlight` purely on those tags.
        user = await self._seed(db)
        matched = await self._names(db, user, status_filter="finalized", search="AR")
        assert "driveway ar tier1" in matched
        assert "kelly closeup" not in matched

    async def test_search_still_matches_ar_inside_a_name_which_is_what_the_pill_is_for(self, db):
        # Deliberate, and the limit of what `q` can do: names are fragment-matched, so "AR" also
        # returns "kitchen argentina" and "starlight bar scene" on their NAMES. That is the same
        # property that makes a half-remembered name findable, and it cannot be both.
        #
        # The precise control is the tag pill, which returns only the AR-tagged job -- see
        # test_a_tag_pill_does_not_match_a_longer_tag_that_starts_with_it. If name noise still
        # bites in practice, word-boundary name matching is the follow-up, not a wider `q`.
        user = await self._seed(db)
        assert await self._names(db, user, status_filter="finalized", search="AR") == {
            "driveway ar tier1",
            "kitchen argentina",
            "starlight bar scene",
        }

    async def test_search_still_matches_a_name_fragment(self, db):
        user = await self._seed(db)
        assert await self._names(db, user, status_filter="finalized", search="kitchen") == {
            "kitchen argentina",
        }

    async def test_search_and_tags_and_together(self, db):
        user = await self._seed(db)
        assert await self._names(db, user, status_filter="finalized", search="driveway", tags=["kelly"]) == {
            "driveway ar tier1",
        }

    async def test_status_filter_still_applies_alongside_tags(self, db):
        # The Videos page only ever asks for finalized work; a pill must not widen that.
        user = await self._seed(db)
        assert await self._names(db, user, status_filter="finalized", tags=["ar"]) == {
            "driveway ar tier1",
        }

    async def test_blank_tags_are_ignored_rather_than_filtering_nothing(self, db):
        user = await self._seed(db)
        assert await self._names(db, user, status_filter="finalized", tags=["  "]) == {
            "driveway ar tier1", "kitchen argentina", "starlight bar scene",
            "kelly closeup", "untagged take",
        }

    async def test_like_wildcards_in_a_tag_are_escaped(self, db):
        # A tag of "%" must not become match-everything.
        user = await self._seed(db)
        assert await self._names(db, user, status_filter="finalized", tags=["%"]) == set()


@pytest.mark.asyncio
class TestTagCountsEndpoint:
    """`/jobs/tag-counts` — the numbers on the pills.

    Counts are scoped to the CURRENT filter, which is what makes the pill row navigable: with
    `kelly` selected, the tags left standing are the ones that co-occur with it, so a dead end is
    visible before it is clicked instead of after.
    """

    async def _counts(self, db, user, **params):
        app.dependency_overrides[get_db] = lambda: db
        app.dependency_overrides[get_current_user] = lambda: user
        try:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                res = await client.get("/jobs/tag-counts", params=params)
            assert res.status_code == 200, res.text
            return {row["tag"]: row["count"] for row in res.json()["items"]}
        finally:
            app.dependency_overrides.clear()

    async def test_counts_every_tag_in_use_under_the_status_filter(self, db):
        user = await TestJobFilter()._seed(db)
        # The Videos page asks for finalized work only, so the pending `depth` job is not counted.
        assert await self._counts(db, user, status="finalized") == {
            "ar": 1, "kelly": 3, "argentina": 1, "starlight": 1,
        }

    async def test_counts_narrow_to_the_selected_tag(self, db):
        user = await TestJobFilter()._seed(db)
        # With kelly selected, only what co-occurs with kelly is offered — and kelly itself
        # survives at the result count, so the pill can be clicked off again.
        assert await self._counts(db, user, status="finalized", tags=["kelly"]) == {
            "kelly": 3, "ar": 1, "argentina": 1,
        }

    async def test_a_route_named_tag_counts_is_not_swallowed_by_jobs_job_id(self, db):
        # /jobs/{job_id} takes a UUID, so a path reaching it first would 422 rather than fall
        # through. This is the regression guard for the declaration order.
        user = await TestJobFilter()._seed(db)
        assert "ar" in await self._counts(db, user, status="finalized")
