"""Tests for drain state surviving a worker re-registration.

The regression these guard: a drain was silently cancelled ~30s after it took effect.
The daemon honours the drain, finishes its segment and exits; the RunPod container then
respawns automatically, the daemon re-registers under the same friendly_name, and
register_worker reset status to "online-idle" and drain_after_jobs to None. The worker
resumed claiming work, so the drain looked like it had been ignored entirely.

Pure logic — no HTTP, no database, matching the house style.
"""

import pytest

from app.routes.workers import reregistered_drain_state


class TestDrainSurvivesReregistration:
    def test_draining_status_is_preserved(self):
        """The bug: this used to come back ("online-idle", None)."""
        assert reregistered_drain_state("draining", None) == ("draining", None)

    def test_pending_countdown_is_preserved(self):
        """drain-after-N is an operator request too, and must survive a restart."""
        assert reregistered_drain_state("online-busy", 3) == ("online-idle", 3)

    @pytest.mark.parametrize("status", ["online-idle", "online-busy", "offline", None])
    def test_non_draining_worker_comes_back_idle(self, status):
        assert reregistered_drain_state(status, None) == ("online-idle", None)

    def test_draining_clears_the_countdown(self):
        """Once status is draining the countdown is spent; keeping it would let a later
        cancel_drain leave a stale countdown that re-drains the worker unexpectedly."""
        assert reregistered_drain_state("draining", 2) == ("draining", None)

    def test_countdown_is_not_decremented_here(self):
        """Only update_status decrements. A restart must not consume a job's worth."""
        _, remaining = reregistered_drain_state("online-busy", 5)
        assert remaining == 5


class TestCancelIsTheOnlyEscape:
    def test_a_drained_worker_stays_drained_across_restarts(self):
        """Restart loop: each cycle must keep returning 'draining', never drift to idle."""
        status, after = "draining", None
        for _ in range(5):
            status, after = reregistered_drain_state(status, after)
        assert (status, after) == ("draining", None)
