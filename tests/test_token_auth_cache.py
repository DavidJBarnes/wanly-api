"""wanly-api#434: a page of thumbnails must not cost one users SELECT per image.

GET /files authenticates a browser by ?token=. Every request used to SELECT the token's user;
150+ thumbnails at once took the whole connection pool and hung the 2 GB box. The signature
and expiry are still checked every time; only "this user exists" is remembered, and only yes.
"""
import uuid

import pytest
from fastapi import HTTPException

from app import auth


class _Db:
    def __init__(self, exists=True):
        self.calls = 0
        self.exists = exists

    async def execute(self, _stmt):
        self.calls += 1
        exists = self.exists

        class _R:
            def scalar_one_or_none(self):
                return uuid.uuid4() if exists else None
        return _R()


class _Req:
    def __init__(self, token):
        self.headers = {}
        self.query_params = {"token": token}


@pytest.fixture(autouse=True)
def _fresh_cache():
    auth._user_seen.clear()
    yield
    auth._user_seen.clear()


@pytest.mark.asyncio
async def test_a_burst_of_thumbnails_asks_the_database_once():
    token = auth.create_access_token(uuid.uuid4())
    db = _Db()
    for _ in range(200):
        await auth.verify_api_key_or_token(_Req(token), db)
    assert db.calls == 1


@pytest.mark.asyncio
async def test_a_missing_user_is_refused_and_never_remembered():
    token = auth.create_access_token(uuid.uuid4())
    db = _Db(exists=False)
    for _ in range(3):
        with pytest.raises(HTTPException) as e:
            await auth.verify_api_key_or_token(_Req(token), db)
        assert e.value.status_code == 401
    assert db.calls == 3


@pytest.mark.asyncio
async def test_a_bad_signature_never_reaches_the_cache_or_the_database():
    db = _Db()
    with pytest.raises(HTTPException):
        await auth.verify_api_key_or_token(_Req("not.a.jwt"), db)
    assert db.calls == 0


@pytest.mark.asyncio
async def test_the_answer_expires(monkeypatch):
    token = auth.create_access_token(uuid.uuid4())
    db = _Db()
    await auth.verify_api_key_or_token(_Req(token), db)
    now = auth.time.monotonic()
    monkeypatch.setattr(auth.time, "monotonic", lambda: now + auth.USER_EXISTS_TTL_S + 1)
    await auth.verify_api_key_or_token(_Req(token), db)
    assert db.calls == 2
