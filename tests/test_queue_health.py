"""Stalled = work exists AND nobody can take it.

The 3090's host rebooted on 2026-08-08 and its container had restart=no. It stayed down for
thirteen hours with four segments queued and was found by hand. Every fact needed to catch it was
already in the database; nothing combined them.
"""

from datetime import datetime, timezone

from app.queue_health import LIVE_STATUSES, assess


class TestStalled:
    def test_work_and_no_workers_is_the_outage(self):
        h = assess(pending_segments=4, worker_statuses=[])
        assert h.stalled
        assert h.summary == "4 segments queued and no workers online"

    def test_offline_workers_do_not_count_as_present(self):
        # The real case: the worker row still existed, marked offline. A naive "are there any
        # workers" check would have seen one and stayed quiet for thirteen hours.
        h = assess(pending_segments=4, worker_statuses=["offline"])
        assert h.stalled
        assert h.live_workers == 0

    def test_queued_work_with_a_busy_worker_is_not_an_outage(self):
        # This is just a queue doing its job. Alarming here would fire constantly.
        assert not assess(pending_segments=12, worker_statuses=["online-busy"]).stalled

    def test_no_workers_and_no_work_is_a_quiet_night(self):
        assert not assess(pending_segments=0, worker_statuses=[]).stalled
        assert not assess(pending_segments=0, worker_statuses=["offline"]).stalled

    def test_a_draining_worker_does_not_keep_the_queue_alive(self):
        # Draining means "finish what you hold, take nothing new", so a queue served only by
        # draining workers is every bit as stalled -- it just has not noticed yet.
        assert "draining" not in LIVE_STATUSES
        assert assess(pending_segments=3, worker_statuses=["draining"]).stalled

    def test_one_live_worker_among_dead_ones_is_enough(self):
        h = assess(pending_segments=9, worker_statuses=["offline", "offline", "online-idle"])
        assert not h.stalled
        assert h.live_workers == 1


class TestSummary:
    def test_singular_reads_correctly(self):
        assert assess(pending_segments=1, worker_statuses=[]).summary.startswith("1 segment ")

    def test_empty_when_healthy(self):
        # The caller renders on truthiness, so a healthy summary must not be a non-empty string.
        assert assess(pending_segments=0, worker_statuses=["online-idle"]).summary == ""

    def test_last_worker_seen_is_carried_through(self):
        seen = datetime(2026, 8, 8, 19, 28, tzinfo=timezone.utc)
        assert assess(pending_segments=4, worker_statuses=["offline"],
                      last_worker_seen=seen).last_worker_seen == seen
