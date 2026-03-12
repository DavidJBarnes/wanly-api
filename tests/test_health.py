"""Tests for the /health endpoint."""

import pytest


class TestHealthEndpoint:
    """Verify the health check endpoint works without any auth."""

    @pytest.mark.asyncio
    async def test_health_returns_200(self):
        """GET /health should return 200 with status ok."""
        from httpx import ASGITransport, AsyncClient
        from app.main import app

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    @pytest.mark.asyncio
    async def test_health_requires_no_auth(self):
        """GET /health should not require any authentication headers."""
        from httpx import ASGITransport, AsyncClient
        from app.main import app

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health")
        assert resp.status_code == 200
