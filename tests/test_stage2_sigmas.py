"""Stage 2 sigmas: the Settings value and what it accepts (console#717)."""

import pytest

from app import stage2_sigmas
from app.ltx_stack import LTX_STACK
from app.models import AppSetting
from app.routes.app_settings import _unpin_defaults
from app.schemas.app_settings import AppSettingsUpdate
from tests.test_character_identity import _claim, _queued, presign  # noqa: F401


@pytest.mark.parametrize("good, stored", [
    ("0.725, 0.4219, 0.0", "0.725, 0.4219, 0.0"),
    ("0.7,0", "0.7, 0"),
    ("  ", ""),
    ("", ""),
])
def test_accepted(good, stored):
    assert AppSettingsUpdate(stage2_sigmas=good).stage2_sigmas == stored


@pytest.mark.parametrize("bad", [
    "0.5",                 # nothing to step between
    "1.5, 0",              # above the schedule's range
    "0.4, 0.7, 0",         # not decreasing
    "0.7, 0.7, 0",         # a zero-length step
    "0.7, 0.4",            # leaves noise in the output
    "1e-1, 0",             # ManualSigmas would misread it
    "0.7, abc, 0",
    "0.7, -0.1, 0",
    ", ".join(["0.5"] * 21),
])
def test_refused(bad):
    with pytest.raises(ValueError):
        AppSettingsUpdate(stage2_sigmas=bad)


def test_omitted_leaves_it_alone():
    assert "stage2_sigmas" not in AppSettingsUpdate().model_dump(exclude_none=True)


def test_the_default_typed_back_in_stores_blank():
    """Compared as numbers: 0.725 and the stack's 0.7250 are the same schedule."""
    assert LTX_STACK["sigmas_stage_2"] == "0.85, 0.7250, 0.4219, 0.0"
    out = _unpin_defaults({"stage2_sigmas": "0.85, 0.725, 0.4219, 0"}, {})
    assert out["stage2_sigmas"] == ""


def test_a_different_schedule_is_kept():
    out = _unpin_defaults({"stage2_sigmas": "0.725, 0.4219, 0.0"}, {})
    assert out["stage2_sigmas"] == "0.725, 0.4219, 0.0"


async def test_blank_or_missing_means_none(db):
    assert await stage2_sigmas.configured_stage2_sigmas(db) is None
    db.add(AppSetting(key="stage2_sigmas", value="  "))
    await db.flush()
    assert await stage2_sigmas.configured_stage2_sigmas(db) is None


async def test_set_is_returned(db):
    db.add(AppSetting(key="stage2_sigmas", value="0.7, 0"))
    await db.flush()
    assert await stage2_sigmas.configured_stage2_sigmas(db) == "0.7, 0"


class TestClaim:
    """What the worker is handed, against the real claim query."""

    @pytest.fixture(autouse=True)
    def _worker(self, presign):
        pass

    async def _claim_with(self, db, setting, blob=None):
        if setting is not None:
            db.add(AppSetting(key="stage2_sigmas", value=setting))
        seg = await _queued(db, blob or {"recipe": "Turn"})
        return seg, (await _claim(db))["ltx_recipe"]

    async def test_nothing_set_sends_nothing(self, db):
        _, recipe = await self._claim_with(db, None)
        assert recipe == {"recipe": "Turn"}

    async def test_set_is_sent_and_recorded_on_the_segment(self, db):
        seg, recipe = await self._claim_with(db, "0.7, 0")
        assert recipe["stage2_sigmas"] == "0.7, 0"
        await db.refresh(seg)
        assert seg.ltx_recipe["stage2_sigmas"] == "0.7, 0"

    async def test_clearing_it_drops_a_value_a_reroll_copied(self, db):
        _, recipe = await self._claim_with(
            db, "", {"recipe": "Turn", "stage2_sigmas": "0.7, 0"})
        assert "stage2_sigmas" not in recipe
