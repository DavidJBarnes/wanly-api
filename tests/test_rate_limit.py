"""Unit tests for login rate limiting.

Verifies that POST /login enforces a per-IP request cap via slowapi.
Uses the app's default LOGIN_RATE_LIMIT (5/minute) and mocks the DB
dependency so no real database connection is needed.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.database import get_db
from app.main import app


async def _mock_get_db():
    """Yield a mock session where user lookup always returns None (→ 401)."""
    db = AsyncMock()
    result = MagicMock()  # scalar_one_or_none() is synchronous
    result.scalar_one_or_none.return_value = None
    db.execute.return_value = result
    yield db


class TestLoginRateLimit:
    @pytest.mark.asyncio
    async def test_login_returns_429_after_limit_exceeded(self):
        """The 6th login attempt within a minute returns 429 Too Many Requests."""
        from httpx import ASGITransport, AsyncClient

        app.dependency_overrides[get_db] = _mock_get_db
        try:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                payload = {"username": "anyone", "password": "anything"}

                # First 5 requests should pass rate limiting (get 401 — bad creds)
                for i in range(5):
                    resp = await client.post("/login", json=payload)
                    assert resp.status_code == 401, (
                        f"Request {i + 1}: expected 401, got {resp.status_code}"
                    )

                # 6th request should be rate-limited before reaching the handler
                resp = await client.post("/login", json=payload)
                assert resp.status_code == 429
        finally:
            app.dependency_overrides.clear()
