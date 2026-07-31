"""Tests for per-clip smashcut playback-speed validation.

Two behaviours matter beyond plain range-checking:

  * An all-1.0 list must collapse to None. The daemon keys its fast stream-copy concat on
    "no speeds requested"; a list of 1.0s would silently force a full re-encode that changes
    nothing, so the cheap path has to survive a UI that always sends its defaults.

  * Length must match the picked clips. The speeds ride in a list parallel to the clip paths,
    so a mismatch is not a cosmetic error — it would misalign every clip after the gap and
    retime the wrong ones.

Pure logic — no HTTP, no database, matching the house style.
"""

import pytest
from fastapi import HTTPException

from app.routes.segments import SMASHCUT_SPEED_MAX, SMASHCUT_SPEED_MIN, normalize_clip_speeds


class TestNoRetimingCollapsesToNone:
    def test_omitted_speeds_stay_none(self):
        assert normalize_clip_speeds(3, None) is None

    def test_all_ones_collapse_to_none(self):
        """The console sends a full list every build; defaults must not cost a re-encode."""
        assert normalize_clip_speeds(3, [1.0, 1.0, 1.0]) is None

    def test_one_non_default_keeps_the_whole_list(self):
        assert normalize_clip_speeds(3, [1.0, 2.0, 1.0]) == [1.0, 2.0, 1.0]


class TestAlignmentWithClips:
    @pytest.mark.parametrize("count,speeds", [
        (3, [1.0, 2.0]),          # too few — every later clip would shift
        (2, [1.0, 2.0, 1.0]),     # too many
        (2, []),                  # empty is not "unspecified"; None is
    ])
    def test_length_mismatch_is_rejected(self, count, speeds):
        with pytest.raises(HTTPException) as e:
            normalize_clip_speeds(count, speeds)
        assert e.value.status_code == 400

    def test_order_is_preserved(self):
        """Position is the only thing binding a speed to its clip."""
        assert normalize_clip_speeds(4, [2.0, 0.5, 1.0, 4.0]) == [2.0, 0.5, 1.0, 4.0]


class TestBounds:
    @pytest.mark.parametrize("bad", [0.0, 0.1, 4.1, 100.0, -1.0])
    def test_out_of_range_is_rejected(self, bad):
        with pytest.raises(HTTPException) as e:
            normalize_clip_speeds(1, [bad])
        assert e.value.status_code == 400

    @pytest.mark.parametrize("edge", [SMASHCUT_SPEED_MIN, SMASHCUT_SPEED_MAX])
    def test_the_bounds_themselves_are_allowed(self, edge):
        assert normalize_clip_speeds(1, [edge]) == [edge]

    def test_slow_motion_is_supported(self):
        """Below 1.0 is the whole point of the lower bound — not just a guard."""
        assert normalize_clip_speeds(2, [0.5, 0.25]) == [0.5, 0.25]

    def test_ints_are_coerced_to_float(self):
        """JSON sends 2, not 2.0; the daemon divides by this value."""
        result = normalize_clip_speeds(1, [2])
        assert result == [2.0]
        assert isinstance(result[0], float)
