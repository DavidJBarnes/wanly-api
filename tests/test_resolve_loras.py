"""Unit tests for LoRA reference resolution.

Tests _resolve_loras() from app/routes/segments.py, which translates
{"lora_id": "<uuid>"} entries into full model metadata for the worker daemon.
The DB session is mocked so no database is required.
"""

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from app.routes.segments import _resolve_loras


def _make_lora(**overrides):
    """Build a mock Lora ORM object with sensible defaults."""
    lora = MagicMock()
    lora.id = overrides.get("id", uuid.uuid4())
    lora.high_file = overrides.get("high_file", "model_v2_high.safetensors")
    lora.high_s3_uri = overrides.get("high_s3_uri", "s3://loras/model_v2_high.safetensors")
    lora.low_file = overrides.get("low_file", "model_v2_low.safetensors")
    lora.low_s3_uri = overrides.get("low_s3_uri", "s3://loras/model_v2_low.safetensors")
    lora.default_high_weight = overrides.get("default_high_weight", 1.0)
    lora.default_low_weight = overrides.get("default_low_weight", 0.8)
    return lora


class TestResolveLoras:
    @pytest.mark.asyncio
    async def test_none_input_passes_through(self):
        """None loras input returns None without touching the DB."""
        db = AsyncMock()
        assert await _resolve_loras(db, None) is None
        db.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_list_passes_through(self):
        """Empty list returns empty list."""
        db = AsyncMock()
        assert await _resolve_loras(db, []) == []

    @pytest.mark.asyncio
    async def test_lora_id_resolved_with_default_weights(self):
        """A lora_id reference is expanded to full metadata using the model's defaults."""
        lora = _make_lora()
        db = AsyncMock()
        db.get.return_value = lora

        result = await _resolve_loras(db, [{"lora_id": str(lora.id)}])

        assert len(result) == 1
        entry = result[0]
        assert entry["lora_id"] == str(lora.id)
        assert entry["high_file"] == "model_v2_high.safetensors"
        assert entry["high_s3_uri"] == "s3://loras/model_v2_high.safetensors"
        assert entry["high_weight"] == 1.0
        assert entry["low_weight"] == 0.8

    @pytest.mark.asyncio
    async def test_custom_weights_override_defaults(self):
        """User-supplied weights take precedence over the LoRA's defaults."""
        lora = _make_lora(default_high_weight=1.0, default_low_weight=0.8)
        db = AsyncMock()
        db.get.return_value = lora

        result = await _resolve_loras(
            db, [{"lora_id": str(lora.id), "high_weight": 0.5, "low_weight": 0.3}]
        )

        assert result[0]["high_weight"] == 0.5
        assert result[0]["low_weight"] == 0.3

    @pytest.mark.asyncio
    async def test_nonexistent_lora_raises_400(self):
        """Referencing an unknown LoRA ID returns a 400 error with the ID in the message."""
        db = AsyncMock()
        db.get.return_value = None
        bad_id = str(uuid.uuid4())

        with pytest.raises(HTTPException) as exc_info:
            await _resolve_loras(db, [{"lora_id": bad_id}])

        assert exc_info.value.status_code == 400
        assert bad_id in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_non_dict_items_pass_through(self):
        """Legacy raw-string LoRA entries are forwarded unchanged."""
        db = AsyncMock()
        result = await _resolve_loras(db, ["my_model.safetensors"])

        assert result == ["my_model.safetensors"]
        db.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_dict_without_lora_id_passes_through(self):
        """A dict entry with no lora_id key is forwarded unchanged (manual config)."""
        db = AsyncMock()
        manual = {"file": "custom.safetensors", "weight": 0.9}
        result = await _resolve_loras(db, [manual])

        assert result == [manual]
        db.get.assert_not_called()
