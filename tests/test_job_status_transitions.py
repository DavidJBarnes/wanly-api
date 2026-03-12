"""Unit tests for the job status state machine.

The transition rules live in app/enums.py:JOB_VALID_TRANSITIONS. These tests
verify the rules as pure data — no HTTP requests, no database, just the
state machine logic itself.
"""

import pytest

from app.enums import JOB_VALID_TRANSITIONS, JobStatus

ALL_STATUSES = set(JobStatus)


def is_transition_allowed(current: str, target: str) -> bool:
    """Check if a user-initiated status transition is allowed."""
    return target in JOB_VALID_TRANSITIONS.get(current, set())


class TestValidTransitions:
    """Every explicitly defined transition should be accepted."""

    @pytest.mark.parametrize("current,target", [
        ("pending", "paused"),
        ("pending", "archived"),
        ("processing", "paused"),
        ("awaiting", "paused"),
        ("awaiting", "finalized"),
        ("awaiting", "archived"),
        ("failed", "paused"),
        ("failed", "archived"),
        ("paused", "pending"),
        ("paused", "processing"),
        ("paused", "awaiting"),
        ("paused", "archived"),
        ("archived", "awaiting"),
    ])
    def test_allowed(self, current, target):
        assert is_transition_allowed(current, target)


class TestInvalidTransitions:
    """Transitions not in the table should be blocked."""

    @pytest.mark.parametrize("current,target", [
        ("pending", "finalized"),       # can't skip to finalized
        ("pending", "processing"),      # only the system does this via claim
        ("processing", "awaiting"),     # system-only, after segment completion
        ("finalized", "pending"),       # terminal state
        ("finalizing", "pending"),      # terminal state
        ("awaiting", "processing"),     # can't revert
        ("failed", "processing"),       # must go through paused first
    ])
    def test_rejected(self, current, target):
        assert not is_transition_allowed(current, target)


class TestTerminalStates:
    """Finalized and finalizing are terminal — no exits allowed."""

    @pytest.mark.parametrize("terminal", ["finalized", "finalizing"])
    def test_no_outbound_transitions(self, terminal):
        for target in ALL_STATUSES:
            assert not is_transition_allowed(terminal, target), (
                f"{terminal} -> {target} should be blocked"
            )


class TestPauseSymmetry:
    def test_every_active_status_can_pause(self):
        """All non-terminal active statuses support pausing."""
        for s in ("pending", "processing", "awaiting", "failed"):
            assert is_transition_allowed(s, "paused"), f"{s} -> paused should work"

    def test_paused_can_resume_to_active_states(self):
        """Paused jobs can resume to the main active statuses."""
        for s in ("pending", "processing", "awaiting"):
            assert is_transition_allowed("paused", s), f"paused -> {s} should work"

    def test_paused_cannot_resume_to_terminal_or_failed(self):
        """Paused cannot jump directly to failed, finalized, or finalizing."""
        for s in ("failed", "finalized", "finalizing"):
            assert not is_transition_allowed("paused", s), (
                f"paused -> {s} should be blocked"
            )


class TestArchiveTransitions:
    """Archive transitions are available from non-active statuses."""

    def test_pending_can_archive(self):
        assert is_transition_allowed("pending", "archived")

    def test_awaiting_can_archive(self):
        assert is_transition_allowed("awaiting", "archived")

    def test_failed_can_archive(self):
        assert is_transition_allowed("failed", "archived")

    def test_paused_can_archive(self):
        assert is_transition_allowed("paused", "archived")

    def test_archived_can_unarchive_to_awaiting(self):
        assert is_transition_allowed("archived", "awaiting")

    def test_processing_cannot_archive(self):
        assert not is_transition_allowed("processing", "archived")
