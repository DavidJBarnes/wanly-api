"""One default pose and one default character (wanly-console#543).

The console preselects them in the New Job and Next Segment modals, so the rule that matters
is "at most one of each". It is held twice: the set route clears the old default in the same
transaction, and a partial unique index refuses a second TRUE row if anything bypasses it.

Run against a real database: the partial index and the two-statement swap are the subject.
"""

import uuid

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.auth import get_current_user
from app.database import get_db
from app.main import app
from app.models import LtxBook, LtxCharacter, LtxRecipe


async def _call(db, method, path, **kw):
    """Call a route over HTTP exactly as the console does."""
    app.dependency_overrides[get_current_user] = lambda: object()
    app.dependency_overrides[get_db] = lambda: db
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            return await getattr(client, method)(path, **kw)
    finally:
        app.dependency_overrides.clear()


async def _characters(db, n: int) -> list[LtxCharacter]:
    rows = [LtxCharacter(id=uuid.uuid4(), name=f"c-{uuid.uuid4().hex[:8]}", char_lora="none",
                         trigger="t") for _ in range(n)]
    db.add_all(rows)
    await db.flush()
    return rows


async def _poses(db, n: int) -> list[LtxRecipe]:
    book = LtxBook(id=uuid.uuid4(), name=f"b-{uuid.uuid4().hex[:8]}")
    db.add(book)
    await db.flush()
    rows = [LtxRecipe(id=uuid.uuid4(), name=f"p{i}", prompt_template="<TRIGGER>, x",
                      content_loras=[], book_id=book.id) for i in range(n)]
    db.add_all(rows)
    await db.flush()
    return rows


async def _defaults(db, model) -> list[uuid.UUID]:
    return list((await db.execute(select(model.id).where(model.is_default))).scalars().all())


# The same behaviour for both tables, so each test runs against each.
KINDS = [
    pytest.param(LtxCharacter, _characters, "/ltx/characters", id="character"),
    pytest.param(LtxRecipe, _poses, "/ltx/recipes", id="pose"),
]


@pytest.mark.parametrize("model,make,base", KINDS)
class TestSettingTheDefault:
    async def test_nothing_is_the_default_until_someone_sets_one(self, db, model, make, base):
        """Existing rows migrate to false, so the modals behave as before #543."""
        await make(db, 2)
        assert await _defaults(db, model) == []

    async def test_setting_marks_the_row_and_returns_it(self, db, model, make, base):
        a, _ = await make(db, 2)
        r = await _call(db, "post", f"{base}/{a.id}/default")
        assert r.status_code == 200, r.text
        assert r.json()["is_default"] is True
        assert await _defaults(db, model) == [a.id]

    async def test_switching_clears_the_previous_default(self, db, model, make, base):
        a, b = await make(db, 2)
        assert (await _call(db, "post", f"{base}/{a.id}/default")).status_code == 200
        r = await _call(db, "post", f"{base}/{b.id}/default")
        assert r.status_code == 200, r.text
        assert await _defaults(db, model) == [b.id]
        await db.refresh(a)
        assert a.is_default is False

    async def test_setting_the_current_default_again_is_harmless(self, db, model, make, base):
        a, _ = await make(db, 2)
        await _call(db, "post", f"{base}/{a.id}/default")
        r = await _call(db, "post", f"{base}/{a.id}/default")
        assert r.status_code == 200, r.text
        assert await _defaults(db, model) == [a.id]

    async def test_clearing_leaves_no_default(self, db, model, make, base):
        a, _ = await make(db, 2)
        await _call(db, "post", f"{base}/{a.id}/default")
        r = await _call(db, "delete", f"{base}/{a.id}/default")
        assert r.status_code == 200, r.text
        assert r.json()["is_default"] is False
        assert await _defaults(db, model) == []

    async def test_clearing_a_row_that_is_not_the_default_leaves_the_real_one(
            self, db, model, make, base):
        """Idempotent per row: a stale toggle in another tab must not un-default somebody
        else's choice."""
        a, b = await make(db, 2)
        await _call(db, "post", f"{base}/{a.id}/default")
        r = await _call(db, "delete", f"{base}/{b.id}/default")
        assert r.status_code == 200, r.text
        assert await _defaults(db, model) == [a.id]

    async def test_deleting_the_default_row_leaves_no_default(self, db, model, make, base):
        a, b = await make(db, 2)
        await _call(db, "post", f"{base}/{a.id}/default")
        r = await _call(db, "delete", f"{base}/{a.id}")
        assert r.status_code == 204, r.text
        assert await _defaults(db, model) == []
        # ...and another row can become the default afterwards.
        assert (await _call(db, "post", f"{base}/{b.id}/default")).status_code == 200
        assert await _defaults(db, model) == [b.id]

    async def test_an_unknown_id_is_a_404(self, db, model, make, base):
        assert (await _call(db, "post", f"{base}/{uuid.uuid4()}/default")).status_code == 404
        assert (await _call(db, "delete", f"{base}/{uuid.uuid4()}/default")).status_code == 404

    async def test_the_index_refuses_a_second_default(self, db, model, make, base):
        """The backstop: anything writing the column directly still cannot make two."""
        a, b = await make(db, 2)
        a.is_default = True
        await db.flush()
        b.is_default = True
        with pytest.raises(IntegrityError):
            await db.flush()
        await db.rollback()

    async def test_patch_cannot_set_it(self, db, model, make, base):
        """Only the default route may, because only it clears the old one."""
        a, _ = await make(db, 2)
        r = await _call(db, "patch", f"{base}/{a.id}", json={"is_default": True})
        assert r.status_code == 200, r.text
        assert await _defaults(db, model) == []


class TestListings:
    async def test_get_recipes_carries_is_default_on_poses_and_characters(self, db):
        c, other_c = await _characters(db, 2)
        p, other_p = await _poses(db, 2)
        await _call(db, "post", f"/ltx/characters/{c.id}/default")
        await _call(db, "post", f"/ltx/recipes/{p.id}/default")

        r = await _call(db, "get", "/recipes")
        assert r.status_code == 200, r.text
        body = r.json()
        chars = {x["id"]: x["is_default"] for x in body["characters"]}
        poses = {x["id"]: x["is_default"] for x in body["poses"]}
        assert chars[str(c.id)] is True and chars[str(other_c.id)] is False
        assert poses[str(p.id)] is True and poses[str(other_p.id)] is False

    async def test_list_characters_carries_is_default(self, db):
        c, other = await _characters(db, 2)
        await _call(db, "post", f"/ltx/characters/{c.id}/default")
        r = await _call(db, "get", "/ltx/characters")
        assert r.status_code == 200, r.text
        flags = {x["id"]: x["is_default"] for x in r.json()}
        assert flags[str(c.id)] is True and flags[str(other.id)] is False

    async def test_a_patched_pose_still_reports_its_default(self, db):
        """PATCH responses carry it too, so an edit in the console never un-stars the row."""
        p, _ = await _poses(db, 2)
        await _call(db, "post", f"/ltx/recipes/{p.id}/default")
        r = await _call(db, "patch", f"/ltx/recipes/{p.id}", json={"frames": 97})
        assert r.status_code == 200, r.text
        assert r.json()["is_default"] is True
