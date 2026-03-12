"""Tests for status enums and transition rules."""

import pytest

from app.enums import JOB_VALID_TRANSITIONS, JobStatus, SegmentStatus, VideoStatus


class TestJobStatusEnum:
    """Verify JobStatus enum has all expected values."""

    def test_all_statuses_present(self):
        expected = {"pending", "processing", "awaiting", "failed", "paused", "finalized", "finalizing", "archived"}
        assert set(JobStatus) == expected

    def test_str_enum_values_match(self):
        assert JobStatus.PENDING == "pending"
        assert JobStatus.ARCHIVED == "archived"

    def test_is_str_subclass(self):
        assert isinstance(JobStatus.PENDING, str)


class TestSegmentStatusEnum:
    def test_all_statuses_present(self):
        expected = {"pending", "claimed", "processing", "completed", "failed"}
        assert set(SegmentStatus) == expected


class TestVideoStatusEnum:
    def test_all_statuses_present(self):
        expected = {"pending", "completed", "failed"}
        assert set(VideoStatus) == expected


class TestTransitionMap:
    """Verify the centralized transition map covers all non-terminal statuses."""

    def test_all_non_terminal_statuses_have_transitions(self):
        non_terminal = {JobStatus.PENDING, JobStatus.PROCESSING, JobStatus.AWAITING,
                        JobStatus.FAILED, JobStatus.PAUSED, JobStatus.ARCHIVED}
        for status in non_terminal:
            assert status in JOB_VALID_TRANSITIONS, f"{status} missing from transition map"
            assert len(JOB_VALID_TRANSITIONS[status]) > 0, f"{status} has no transitions"

    def test_terminal_statuses_not_in_map(self):
        assert JobStatus.FINALIZED not in JOB_VALID_TRANSITIONS
        assert JobStatus.FINALIZING not in JOB_VALID_TRANSITIONS

    def test_transition_values_are_valid_statuses(self):
        all_statuses = set(JobStatus)
        for source, targets in JOB_VALID_TRANSITIONS.items():
            assert source in all_statuses, f"Source {source} not a valid status"
            for target in targets:
                assert target in all_statuses, f"Target {target} not a valid status"
