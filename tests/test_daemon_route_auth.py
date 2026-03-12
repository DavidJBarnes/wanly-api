"""Tests verifying API key auth on daemon-facing routes.

All 5 previously-unprotected daemon routes now require X-API-Key header.
"""

import pytest

from app.main import app


class TestDaemonRoutesRequireApiKey:
    """Verify that daemon-facing routes reject requests without a valid API key."""

    @pytest.mark.asyncio
    async def test_list_segments_without_api_key_returns_401(self):
        """GET /segments requires X-API-Key."""
        from httpx import ASGITransport, AsyncClient

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/segments", params={"worker_id": "00000000-0000-0000-0000-000000000001"})
        assert resp.status_code in (401, 403)

    @pytest.mark.asyncio
    async def test_claim_next_without_api_key_returns_401(self):
        """GET /segments/next requires X-API-Key."""
        from httpx import ASGITransport, AsyncClient

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/segments/next", params={"worker_id": "00000000-0000-0000-0000-000000000001"})
        assert resp.status_code in (401, 403)

    @pytest.mark.asyncio
    async def test_update_segment_without_api_key_returns_401(self):
        """PATCH /segments/{id} requires X-API-Key."""
        from httpx import ASGITransport, AsyncClient

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.patch(
                "/segments/00000000-0000-0000-0000-000000000001",
                json={"status": "processing"},
            )
        assert resp.status_code in (401, 403)

    @pytest.mark.asyncio
    async def test_download_file_without_api_key_returns_401(self):
        """GET /files requires X-API-Key."""
        from httpx import ASGITransport, AsyncClient

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/files", params={"path": "s3://bucket/key"})
        assert resp.status_code in (401, 403)

    @pytest.mark.asyncio
    async def test_upload_segment_without_api_key_returns_401(self):
        """POST /segments/{id}/upload requires X-API-Key."""
        from httpx import ASGITransport, AsyncClient

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/segments/00000000-0000-0000-0000-000000000001/upload",
                files={
                    "video": ("test.mp4", b"fake-video", "video/mp4"),
                    "last_frame": ("test.png", b"fake-frame", "image/png"),
                },
            )
        assert resp.status_code in (401, 403)


class TestDaemonRoutesWithInvalidApiKey:
    """Verify that an invalid API key is rejected."""

    @pytest.mark.asyncio
    async def test_invalid_api_key_returns_401(self):
        """An incorrect X-API-Key header should be rejected."""
        from httpx import ASGITransport, AsyncClient

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/segments/next",
                params={"worker_id": "00000000-0000-0000-0000-000000000001"},
                headers={"X-API-Key": "wrong-key"},
            )
        assert resp.status_code == 401
