"""Unit tests for wildcard placeholder resolution in segment prompts.

Tests _resolve_wildcards() from app/routes/segments.py, which substitutes
<name> placeholders in prompts with random choices from the Wildcard table.
The DB session is mocked so no database is required.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.routes.segments import _resolve_wildcards


def _make_wildcard(name: str, options: list[str]):
    """Build a mock Wildcard ORM object."""
    wc = MagicMock()
    wc.name = name
    wc.options = options
    return wc


def _mock_db(wildcards: list):
    """Build an AsyncSession mock whose execute() returns the given wildcards."""
    db = AsyncMock()
    scalars = MagicMock()
    scalars.all.return_value = wildcards
    result = MagicMock()
    result.scalars.return_value = scalars
    db.execute.return_value = result
    return db


class TestResolveWildcards:
    @pytest.mark.asyncio
    async def test_no_placeholders_returns_prompt_unchanged(self):
        """A plain prompt with no <angle brackets> comes back as-is, template=None."""
        db = AsyncMock()
        resolved, template = await _resolve_wildcards(db, "a sunset over the ocean")

        assert resolved == "a sunset over the ocean"
        assert template is None
        db.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_single_wildcard_is_substituted(self):
        """A <style> placeholder is replaced with one of its options."""
        # Use a single option so the outcome is deterministic
        db = _mock_db([_make_wildcard("style", ["cinematic"])])

        resolved, template = await _resolve_wildcards(db, "a <style> portrait")

        assert resolved == "a cinematic portrait"
        assert template == "a <style> portrait"

    @pytest.mark.asyncio
    async def test_unknown_wildcard_stays_in_prompt(self):
        """A placeholder with no matching DB record is left unreplaced."""
        db = _mock_db([])  # DB has no wildcards

        resolved, template = await _resolve_wildcards(db, "a <nonexistent> landscape")

        assert "<nonexistent>" in resolved
        assert template == "a <nonexistent> landscape"

    @pytest.mark.asyncio
    async def test_multiple_different_wildcards(self):
        """Two distinct placeholders are each resolved from their own options."""
        db = _mock_db([
            _make_wildcard("style", ["anime"]),
            _make_wildcard("color", ["blue"]),
        ])

        resolved, template = await _resolve_wildcards(
            db, "a <style> scene with <color> tones"
        )

        assert resolved == "a anime scene with blue tones"
        assert template == "a <style> scene with <color> tones"

    @pytest.mark.asyncio
    async def test_duplicate_placeholder_replaced_everywhere(self):
        """All occurrences of the same <name> get the same replacement."""
        db = _mock_db([_make_wildcard("mood", ["joyful"])])

        resolved, _ = await _resolve_wildcards(
            db, "<mood> person in a <mood> setting"
        )

        assert resolved == "joyful person in a joyful setting"
