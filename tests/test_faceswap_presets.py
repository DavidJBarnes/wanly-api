"""Unit tests for GET /faceswap/presets endpoint.

Mocks the S3 client so no AWS credentials are required.
Uses FastAPI dependency overrides to bypass auth DB queries.
"""

import uuid
from unittest.mock import MagicMock, patch

import pytest

from app.auth import get_current_user
from app.main import app
from app.models import User

_fake_user = User(
    id=uuid.uuid4(), username="testuser",
    password_hash="x",
)


class TestFaceswapPresetsAuth:
    """Verify that GET /faceswap/presets rejects unauthenticated requests."""

    @pytest.mark.asyncio
    async def test_without_token_returns_401(self):
        from httpx import ASGITransport, AsyncClient

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/faceswap/presets")
        assert resp.status_code == 401


class TestFaceswapPresetsList:
    """Verify the endpoint returns correctly shaped data when S3 has objects."""

    def _mock_s3(self, contents, is_truncated=False, next_token=None):
        """Return a patcher that stubs _get_client().list_objects_v2."""
        mock_client = MagicMock()
        resp = {"Contents": contents, "IsTruncated": is_truncated}
        if next_token:
            resp["NextContinuationToken"] = next_token
        mock_client.list_objects_v2.return_value = resp
        return patch("app.routes.faceswap._get_client", return_value=mock_client)

    def setup_method(self):
        app.dependency_overrides[get_current_user] = lambda: _fake_user

    def teardown_method(self):
        app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_returns_presets_with_relative_thumbnail_url(self):
        from httpx import ASGITransport, AsyncClient

        s3_objects = [
            {"Key": "celebrity.png", "Size": 1024},
            {"Key": "subfolder/other.jpg", "Size": 2048},
        ]

        with self._mock_s3(s3_objects):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/faceswap/presets")

        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2

        first = data[0]
        assert first["key"] == "celebrity.png"
        assert first["name"] == "celebrity"
        assert first["url"] == "s3://wanly-faces/celebrity.png"
        assert first["thumbnail_url"].startswith("/files?path=")
        # Must NOT contain a host — relative path only
        assert "http" not in first["thumbnail_url"]

        second = data[1]
        assert second["key"] == "subfolder/other.jpg"
        assert second["name"] == "other"

    @pytest.mark.asyncio
    async def test_filters_out_folder_markers_and_non_images(self):
        from httpx import ASGITransport, AsyncClient

        s3_objects = [
            {"Key": "faces/", "Size": 0},            # folder marker
            {"Key": "readme.txt", "Size": 100},       # non-image
            {"Key": "face.png", "Size": 5000},         # valid
            {"Key": "empty.jpg", "Size": 0},           # 0-byte
        ]

        with self._mock_s3(s3_objects):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/faceswap/presets")

        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["key"] == "face.png"

    @pytest.mark.asyncio
    async def test_empty_bucket_returns_empty_list(self):
        from httpx import ASGITransport, AsyncClient

        with self._mock_s3([]):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/faceswap/presets")

        assert resp.status_code == 200
        assert resp.json() == []

    @pytest.mark.asyncio
    async def test_s3_failure_returns_502(self):
        from httpx import ASGITransport, AsyncClient

        mock_client = MagicMock()
        mock_client.list_objects_v2.side_effect = Exception("AccessDenied")

        with patch("app.routes.faceswap._get_client", return_value=mock_client):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/faceswap/presets")

        assert resp.status_code == 502

    @pytest.mark.asyncio
    async def test_handles_s3_pagination(self):
        from httpx import ASGITransport, AsyncClient

        page1 = [{"Key": "a.png", "Size": 100}]
        page2 = [{"Key": "b.jpg", "Size": 200}]

        mock_client = MagicMock()
        mock_client.list_objects_v2.side_effect = [
            {"Contents": page1, "IsTruncated": True, "NextContinuationToken": "tok2"},
            {"Contents": page2, "IsTruncated": False},
        ]

        with patch("app.routes.faceswap._get_client", return_value=mock_client):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/faceswap/presets")

        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        assert data[0]["key"] == "a.png"
        assert data[1]["key"] == "b.jpg"
        # Verify pagination token was used
        assert mock_client.list_objects_v2.call_count == 2
