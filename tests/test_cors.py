"""Unit tests for CORS configuration.

Verifies that the middleware only allows explicitly configured origins.
"""

import pytest


class TestCorsPolicy:
    @pytest.mark.asyncio
    async def test_wildcard_origin_not_reflected(self):
        """An arbitrary Origin header does not receive an Access-Control-Allow-Origin echo.

        CORS_ORIGINS defaults to empty in tests, so no origin should be allowed.
        """
        from httpx import ASGITransport, AsyncClient
        from app.main import app

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.options(
                "/login",
                headers={
                    "Origin": "https://evil.com",
                    "Access-Control-Request-Method": "POST",
                },
            )
        assert "access-control-allow-origin" not in resp.headers
