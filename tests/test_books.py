"""Books: poses stop being one flat list and get a shelf (wanly-api#320).

A pose belongs to exactly one book. The interesting behaviours are the ones that are easy to
get wrong and silent when they are:

  * a pose created by a caller who has never heard of books still lands somewhere — the route
    defaults it, so the NOT NULL column never becomes a reason a create fails;
  * two books may hold a pose of the same name, because uniqueness is per book now;
  * a populated book cannot be deleted: 409 with a message, and RESTRICT underneath it;
  * the migration files the existing poses by checkpoint family, not into one bucket, and
    leaves nothing orphaned.

Run against a real database: the FK, the composite unique and the backfill are the subject.
"""

import importlib.util
import uuid
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.exc import IntegrityError

from app.auth import get_current_user
from app.database import get_db
from app.main import app
from app.models import LtxBook, LtxRecipe, User
from app.routes.ltx_recipes import DEFAULT_BOOK_NAME


async def _user(db) -> User:
    user = User(username=str(uuid.uuid4()), password_hash="x")
    db.add(user)
    await db.flush()
    return user


async def _book(db, name: str) -> LtxBook:
    b = LtxBook(id=uuid.uuid4(), name=name)
    db.add(b)
    await db.flush()
    return b


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


class TestBackfillRule:
    """The mapping that decides which book an existing pose lands in. Frozen in the migration,
    so these test the migration's own function rather than a re-implementation of it."""

    @staticmethod
    def _migration():
        path = Path("alembic/versions/100_books.py")
        spec = importlib.util.spec_from_file_location("m100", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_a_10eros_checkpoint_goes_to_10eros(self):
        m = self._migration()
        assert m._book_for("10Eros_v1.5_bf16.safetensors")[0] == "10eros"
        assert m._book_for("10eros")[0] == "10eros"

    def test_a_sulphur_checkpoint_goes_to_sulphur(self):
        m = self._migration()
        assert m._book_for("sulphur_distill.safetensors")[0] == "sulphur"

    def test_matching_is_case_insensitive(self):
        m = self._migration()
        assert m._book_for("SULPHUR_v2.safetensors")[0] == "sulphur"
        assert m._book_for("10EROS_v1.5_bf16")[0] == "10eros"

    def test_a_null_checkpoint_is_the_stack_default_and_lands_in_10eros(self):
        """NULL means "use the stack's value", which is 10Eros_v1.5_bf16. Those poses are
        10Eros poses that never said so, not a category of their own."""
        m = self._migration()
        assert m._book_for(None)[0] == "10eros"
        assert m._book_for("")[0] == "10eros"
        assert m._book_for("   ")[0] == "10eros"

    def test_an_unknown_checkpoint_gets_its_own_book_without_the_extension(self):
        """A base model nobody anticipated is filed under its own name, so it is visible and
        can be renamed, rather than silently filed in a book it does not belong to."""
        m = self._migration()
        name, source = m._book_for("Foo_v3.safetensors")
        assert name == "Foo_v3"
        assert source == "Foo_v3.safetensors"

    def test_the_default_book_name_matches_the_routes(self):
        """If these drift, a migrated database has no book the route will default to and the
        first pose create 409s on a database that looks fine."""
        assert self._migration()._DEFAULT_BOOK == DEFAULT_BOOK_NAME


class TestCreateDefaultsToABook:
    async def test_a_create_with_no_book_id_lands_in_the_default_book(self, db):
        await _book(db, DEFAULT_BOOK_NAME)
        r = await _call(db, "post", "/ltx/recipes", json={
            "name": "no book named",
            "prompt_template": "<TRIGGER>, standing",
            "content_loras": [],
        })
        assert r.status_code == 201, r.text
        body = r.json()
        assert body["book_name"] == DEFAULT_BOOK_NAME

    async def test_the_default_falls_back_to_the_first_book_by_name(self, db):
        """The default name is absent, so the pose lands in whichever book exists. A create
        must not be refused for a database whose books were rearranged."""
        await _book(db, "sulphur")
        r = await _call(db, "post", "/ltx/recipes", json={
            "name": "orphan attempt",
            "prompt_template": "<TRIGGER>, standing",
            "content_loras": [],
        })
        assert r.status_code == 201, r.text
        assert r.json()["book_name"] == "sulphur"

    async def test_with_no_books_at_all_it_is_a_409_not_a_500(self, db):
        """Nothing to file against is a real conflict the console can explain, not a crash.
        The route deliberately does not auto-create a book: that would make book management
        invisible."""
        r = await _call(db, "post", "/ltx/recipes", json={
            "name": "nowhere",
            "prompt_template": "<TRIGGER>, standing",
            "content_loras": [],
        })
        assert r.status_code == 409, r.text
        assert "book" in r.json()["detail"].lower()

    async def test_an_explicit_book_id_is_honoured(self, db):
        b = await _book(db, "sulphur")
        r = await _call(db, "post", "/ltx/recipes", json={
            "name": "filed",
            "prompt_template": "<TRIGGER>, standing",
            "content_loras": [],
            "book_id": str(b.id),
        })
        assert r.status_code == 201, r.text
        assert r.json()["book_name"] == "sulphur"

    async def test_an_unknown_book_id_is_a_404(self, db):
        await _book(db, DEFAULT_BOOK_NAME)
        r = await _call(db, "post", "/ltx/recipes", json={
            "name": "filed nowhere real",
            "prompt_template": "<TRIGGER>, standing",
            "content_loras": [],
            "book_id": str(uuid.uuid4()),
        })
        assert r.status_code == 404, r.text


class TestUniquePerBook:
    async def test_the_same_pose_name_may_exist_in_two_books(self, db):
        a = await _book(db, "10eros")
        b = await _book(db, "sulphur")
        for book in (a, b):
            r = await _call(db, "post", "/ltx/recipes", json={
                "name": "mirrored",
                "prompt_template": "<TRIGGER>, standing",
                "content_loras": [],
                "book_id": str(book.id),
            })
            assert r.status_code == 201, r.text

    async def test_the_same_pose_name_in_the_same_book_is_a_409(self, db):
        a = await _book(db, "10eros")
        body = {"name": "dupe", "prompt_template": "<TRIGGER>, standing",
                "content_loras": [], "book_id": str(a.id)}
        assert (await _call(db, "post", "/ltx/recipes", json=body)).status_code == 201
        assert (await _call(db, "post", "/ltx/recipes", json=body)).status_code == 409


class TestMovingAPose:
    async def test_a_patch_can_move_a_pose_between_books(self, db):
        a = await _book(db, "10eros")
        b = await _book(db, "sulphur")
        r = await _call(db, "post", "/ltx/recipes", json={
            "name": "movable", "prompt_template": "<TRIGGER>, standing",
            "content_loras": [], "book_id": str(a.id),
        })
        rid = r.json()["id"]

        r = await _call(db, "patch", f"/ltx/recipes/{rid}", json={"book_id": str(b.id)})
        assert r.status_code == 200, r.text
        assert r.json()["book_name"] == "sulphur"

    async def test_a_patch_without_a_book_leaves_it_alone(self, db):
        a = await _book(db, "10eros")
        r = await _call(db, "post", "/ltx/recipes", json={
            "name": "stay put", "prompt_template": "<TRIGGER>, standing",
            "content_loras": [], "book_id": str(a.id),
        })
        rid = r.json()["id"]
        r = await _call(db, "patch", f"/ltx/recipes/{rid}", json={"frames": 100})
        assert r.status_code == 200, r.text
        assert r.json()["book_name"] == "10eros"


class TestBookCrud:
    async def test_create_list_and_count(self, db):
        b = await _book(db, "10eros")
        await _book(db, "sulphur")
        from app.models import LtxRecipe as R
        db.add(R(id=uuid.uuid4(), name="p", prompt_template="<TRIGGER>, x",
                 content_loras=[], book_id=b.id))
        await db.flush()

        r = await _call(db, "get", "/ltx/books")
        assert r.status_code == 200, r.text
        by_name = {x["name"]: x for x in r.json()}
        assert by_name["10eros"]["recipe_count"] == 1
        assert by_name["sulphur"]["recipe_count"] == 0

    async def test_a_duplicate_name_is_a_409(self, db):
        r = await _call(db, "post", "/ltx/books", json={"name": "unique"})
        assert r.status_code == 201, r.text
        assert (await _call(db, "post", "/ltx/books", json={"name": "unique"})).status_code == 409

    async def test_a_rename_to_an_existing_name_is_a_409(self, db):
        await _book(db, "taken")
        b = await _book(db, "free")
        r = await _call(db, "patch", f"/ltx/books/{b.id}", json={"name": "taken"})
        assert r.status_code == 409, r.text

    async def test_an_empty_book_deletes(self, db):
        b = await _book(db, "empty")
        r = await _call(db, "delete", f"/ltx/books/{b.id}")
        assert r.status_code == 204, r.text

    async def test_a_populated_book_refuses_to_delete_with_a_message(self, db):
        b = await _book(db, "full")
        db.add(LtxRecipe(id=uuid.uuid4(), name="p", prompt_template="<TRIGGER>, x",
                         content_loras=[], book_id=b.id))
        await db.flush()
        r = await _call(db, "delete", f"/ltx/books/{b.id}")
        assert r.status_code == 409, r.text
        assert "1" in r.json()["detail"]

    async def test_a_missing_book_is_a_404(self, db):
        r = await _call(db, "delete", f"/ltx/books/{uuid.uuid4()}")
        assert r.status_code == 404, r.text


class TestDatabaseRefusesTheWrongThings:
    """The route's checks are the message; these are the backstop underneath them."""

    async def test_the_database_refuses_a_populated_book_delete(self, db):
        b = await _book(db, "restricted")
        db.add(LtxRecipe(id=uuid.uuid4(), name="p", prompt_template="<TRIGGER>, x",
                         content_loras=[], book_id=b.id))
        await db.flush()
        await db.delete(b)
        with pytest.raises(IntegrityError):
            await db.flush()
        await db.rollback()

    async def test_a_pose_cannot_be_filed_under_a_book_that_does_not_exist(self, db):
        with pytest.raises(IntegrityError):
            db.add(LtxRecipe(id=uuid.uuid4(), name="p", prompt_template="<TRIGGER>, x",
                             content_loras=[], book_id=uuid.uuid4()))
            await db.flush()
        await db.rollback()


class TestFiltering:
    async def test_the_book_filter_returns_only_that_books_poses(self, db):
        a = await _book(db, "10eros")
        b = await _book(db, "sulphur")
        for book, name in ((a, "ten"), (b, "sul")):
            db.add(LtxRecipe(id=uuid.uuid4(), name=name, prompt_template="<TRIGGER>, x",
                             content_loras=[], book_id=book.id))
        await db.flush()

        r = await _call(db, "get", "/recipes", params={"book_id": str(a.id)})
        assert r.status_code == 200, r.text
        assert [p["name"] for p in r.json()["poses"]] == ["ten"]
        # The books list is still whole: the filter narrows poses, not the shelf list the
        # dropdown needs.
        assert {x["name"] for x in r.json()["books"]} == {"10eros", "sulphur"}

    async def test_an_unknown_book_filter_is_an_empty_list_not_a_404(self, db):
        """The console may hold an id for a book someone just deleted; that should show an
        empty page, not a failed request."""
        await _book(db, "10eros")
        r = await _call(db, "get", "/recipes", params={"book_id": str(uuid.uuid4())})
        assert r.status_code == 200, r.text
        assert r.json()["poses"] == []
