"""Tests for the sweep that marks silent workers offline (#161).

This code had no tests at all, and it silently reverted an operator's drain.

Sequence that happened for real: a worker went quiet for 240s while identity scoring blocked the
daemon's event loop (wanly-gpu-daemon#117). The sweep flipped `draining` to `offline`. When
heartbeats resumed, `heartbeat()` saw a previously-offline worker and set it back to
`online-idle` — so it went straight back to claiming work, with no record that a drain had ever
been requested.

The daemon-side cause is fixed, but a drain has to survive silence for ANY reason: network
partition, GPU hang, crash. These pin that.

The suite has no database fixture and the models use JSONB, which SQLite cannot compile, so
these assert the compiled predicate rather than executing it. That is a real limitation and it
is why the rule was extracted into offline_sweep_conditions() — the test and the sweep now read
from one definition instead of two that can drift.
"""

from datetime import datetime, timedelta, timezone

from sqlalchemy import update

from app.config import settings
from app.heartbeat_monitor import offline_sweep_conditions
from app.models import Worker


def _sql(cutoff=None) -> str:
    cutoff = cutoff or datetime.now(timezone.utc) - timedelta(seconds=120)
    stmt = update(Worker).where(*offline_sweep_conditions(cutoff)).values(status="offline")
    return str(stmt.compile(compile_kwargs={"literal_binds": False}))


class TestDrainSurvivesSilence:
    def test_draining_workers_are_excluded(self):
        """The bug. Without this the drain is lost the moment a heartbeat is late."""
        sql = _sql()
        assert "status != " in sql
        # Both exclusions must be present — offline AND draining.
        assert sql.count("status !=") >= 2, (
            "a draining worker must not be swept to offline; heartbeat() would then reset it to "
            "online-idle and it would resume claiming"
        )

    def test_the_rule_has_one_definition(self):
        """Re-typing the predicate in a test asserts what someone wished were true."""
        import inspect

        from app import heartbeat_monitor

        src = inspect.getsource(heartbeat_monitor.heartbeat_monitor)
        assert "offline_sweep_conditions(cutoff)" in src, (
            "the sweep must build its WHERE from the shared predicate, or this test drifts "
            "away from the code it is meant to protect"
        )


class TestSweepStillWorks:
    def test_silence_is_still_what_triggers_it(self):
        sql = _sql()
        assert "last_heartbeat <" in sql

    def test_already_offline_workers_are_skipped(self):
        """Otherwise the sweep rewrites the same rows forever and logs them each pass."""
        conditions = offline_sweep_conditions(datetime.now(timezone.utc))
        rendered = " ".join(str(c) for c in conditions)
        assert "status !=" in rendered

    def test_threshold_is_several_missed_beats(self):
        """The daemon heartbeats every 30s. Too tight and a slow request marks a live worker
        dead; the post-sampling tail already pushed past 120s once."""
        assert settings.heartbeat_offline_seconds >= 90, "fewer than 3 missed beats is twitchy"
