"""Tests for app settings: schemas, response conversion, and endpoint auth."""

import pytest

from app.schemas.app_settings import AppSettingsResponse, AppSettingsUpdate


class TestAppSettingsSchemas:
    """Validate the Pydantic schemas for settings."""

    def test_response_includes_negative_prompt(self):
        """AppSettingsResponse must include negative_prompt as a string."""
        resp = AppSettingsResponse(
            cfg_high=1.0,
            cfg_low=1.0,
            lightx2v_strength_high=2.0,
            lightx2v_strength_low=1.0,
            negative_prompt="bad quality, blurry",
        )
        assert resp.negative_prompt == "bad quality, blurry"

    def test_response_requires_negative_prompt(self):
        """AppSettingsResponse should fail without negative_prompt."""
        with pytest.raises(Exception):
            AppSettingsResponse(
                cfg_high=1.0,
                cfg_low=1.0,
                lightx2v_strength_high=2.0,
                lightx2v_strength_low=1.0,
            )

    def test_update_negative_prompt_optional(self):
        """AppSettingsUpdate allows negative_prompt to be omitted."""
        update = AppSettingsUpdate()
        assert update.negative_prompt is None

    def test_update_with_negative_prompt(self):
        """AppSettingsUpdate accepts a negative_prompt string."""
        update = AppSettingsUpdate(negative_prompt="ugly, deformed")
        assert update.negative_prompt == "ugly, deformed"

    def test_update_negative_prompt_excluded_when_none(self):
        """exclude_none should drop negative_prompt when not set."""
        update = AppSettingsUpdate(cfg_high=1.5)
        dumped = update.model_dump(exclude_none=True)
        assert "negative_prompt" not in dumped
        assert dumped["cfg_high"] == 1.5

    def test_update_negative_prompt_included_when_set(self):
        """exclude_none should keep negative_prompt when explicitly set."""
        update = AppSettingsUpdate(negative_prompt="blurry")
        dumped = update.model_dump(exclude_none=True)
        assert dumped["negative_prompt"] == "blurry"

    def test_update_empty_string_is_not_none(self):
        """An empty string negative_prompt should be preserved (not treated as None)."""
        update = AppSettingsUpdate(negative_prompt="")
        dumped = update.model_dump(exclude_none=True)
        assert "negative_prompt" in dumped
        assert dumped["negative_prompt"] == ""


class TestToResponse:
    """Test the _to_response helper in the settings route."""

    def test_to_response_includes_negative_prompt(self):
        from app.routes.app_settings import _to_response

        settings = {
            "cfg_high": "1.5",
            "cfg_low": "1.0",
            "lightx2v_strength_high": "2.0",
            "lightx2v_strength_low": "1.0",
            "negative_prompt": "ugly, blurry",
        }
        resp = _to_response(settings)
        assert resp.negative_prompt == "ugly, blurry"
        assert resp.cfg_high == 1.5

    def test_to_response_empty_negative_prompt(self):
        from app.routes.app_settings import _to_response

        settings = {
            "cfg_high": "1",
            "cfg_low": "1",
            "lightx2v_strength_high": "2.0",
            "lightx2v_strength_low": "1.0",
            "negative_prompt": "",
        }
        resp = _to_response(settings)
        assert resp.negative_prompt == ""


class TestDefaultsIncludeNegativePrompt:
    """Verify the _DEFAULTS dict has the negative_prompt key."""

    def test_defaults_has_negative_prompt(self):
        from app.routes.app_settings import _DEFAULTS

        assert "negative_prompt" in _DEFAULTS


class TestSettingsEndpointAuth:
    """Verify settings endpoints require authentication."""

    @pytest.mark.asyncio
    async def test_get_settings_without_auth_returns_401(self):
        from httpx import ASGITransport, AsyncClient
        from app.main import app

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/settings")
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_put_settings_without_auth_returns_401(self):
        from httpx import ASGITransport, AsyncClient
        from app.main import app

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.put("/settings", json={"negative_prompt": "test"})
        assert resp.status_code == 401


class TestSegmentClaimResponseSchema:
    """Verify SegmentClaimResponse includes negative_prompt."""

    def _make_claim_kwargs(self, **overrides):
        from uuid import uuid4

        defaults = dict(
            id=uuid4(),
            job_id=uuid4(),
            index=0,
            prompt="test prompt",
            duration_seconds=5.0,
            speed=1.0,
            start_image=None,
            loras=None,
            faceswap_enabled=False,
            faceswap_method=None,
            faceswap_source_type=None,
            faceswap_image=None,
            faceswap_faces_order=None,
            faceswap_faces_index=None,
            width=832,
            height=480,
            fps=30,
            seed=42,
        )
        defaults.update(overrides)
        return defaults

    def test_segment_claim_response_has_negative_prompt(self):
        from app.schemas.segments import SegmentClaimResponse

        resp = SegmentClaimResponse(**self._make_claim_kwargs(negative_prompt="ugly, blurry"))
        assert resp.negative_prompt == "ugly, blurry"

    def test_segment_claim_response_negative_prompt_optional(self):
        from app.schemas.segments import SegmentClaimResponse

        resp = SegmentClaimResponse(**self._make_claim_kwargs())
        assert resp.negative_prompt is None
