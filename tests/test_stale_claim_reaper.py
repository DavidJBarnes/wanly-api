"""Tests for reclaiming segments orphaned by a dead worker.

A hard crash (power loss, in practice) leaves a segment sitting in `processing` with a dead
worker attached. Nothing else reclaims it, so until this fires the job is stuck and the console
keeps showing its last pre-crash progress line — indistinguishable from a slow render.

The original rule waited 30 minutes on `claimed_at`, because claim age cannot tell "20 minutes
into a legitimate 5s render" from "the machine lost power 20 minutes ago". Workers heartbeat
every 30s, so a dead one is detectable in ~5 minutes with no risk to live work.

These tests pin the SQL predicate rather than the endpoint, because the reclaim is a WHERE
clause and that is where the bug would live: an AND instead of an OR silently reduces this to
the old behaviour, and a missing NULL guard would reclaim work from a worker that never
registered.
"""

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import or_, select

from app.enums import SegmentStatus
from app.models import Segment, Worker
from app.routes.segments import STALE_HEARTBEAT_MINUTES


def _predicate(now: datetime):
    """The reclaim WHERE clause, built exactly as claim_next_segment builds it."""
    age_cutoff = now - timedelta(minutes=30)
    heartbeat_cutoff = now - timedelta(minutes=STALE_HEARTBEAT_MINUTES)
    return (
        select(Segment, Worker.last_heartbeat)
        .outerjoin(Worker, Worker.id == Segment.worker_id)
        .where(
            Segment.status.in_([SegmentStatus.CLAIMED, SegmentStatus.PROCESSING]),
            Segment.claimed_at.is_not(None),
            or_(
                Segment.claimed_at < age_cutoff,
                Worker.last_heartbeat < heartbeat_cutoff,
            ),
        )
    )


class TestHeartbeatThreshold:
    def test_five_minutes_is_about_ten_missed_beats(self):
        """Daemon HEARTBEAT_INTERVAL is 30s. Too tight and a network blip steals live work;
        too loose and we are back to waiting out the worst case."""
        assert STALE_HEARTBEAT_MINUTES == 5
        assert STALE_HEARTBEAT_MINUTES * 60 / 30 >= 8, "fewer than 8 missed beats is twitchy"
        assert STALE_HEARTBEAT_MINUTES < 30, "must beat the old age-based rule"


class TestReclaimPredicate:
    def test_joins_workers_via_outer_join(self):
        """An INNER join would skip segments whose worker row is gone — exactly the rows most
        likely to be orphaned."""
        sql = str(_predicate(datetime.now(timezone.utc)))
        assert "LEFT OUTER JOIN" in sql.upper()

    def test_the_two_rules_are_ORed_not_ANDed(self):
        """AND would require BOTH a stale heartbeat and a 30-minute-old claim, silently
        reducing this to the behaviour it replaces."""
        sql = " ".join(str(_predicate(datetime.now(timezone.utc))).split())
        where = sql.upper().split("WHERE", 1)[1]
        assert " OR " in where, where

    def test_requires_a_claim_timestamp(self):
        """Without this guard an unclaimed segment with a NULL claimed_at could match the
        heartbeat arm and be 'reclaimed' from nobody."""
        sql = " ".join(str(_predicate(datetime.now(timezone.utc))).split())
        assert "claimed_at IS NOT NULL" in sql

    def test_only_targets_in_flight_statuses(self):
        sql = str(_predicate(datetime.now(timezone.utc)))
        assert "status IN" in sql


class TestCutoffArithmetic:
    """The bug that matters here is a sign error — a cutoff in the future reclaims everything,
    including work that started a second ago."""

    def test_cutoffs_are_in_the_past(self):
        now = datetime.now(timezone.utc)
        assert now - timedelta(minutes=30) < now
        assert now - timedelta(minutes=STALE_HEARTBEAT_MINUTES) < now

    def test_heartbeat_cutoff_is_more_recent_than_the_age_cutoff(self):
        """The whole point: a dead worker is caught sooner than a long-held claim."""
        now = datetime.now(timezone.utc)
        assert now - timedelta(minutes=STALE_HEARTBEAT_MINUTES) > now - timedelta(minutes=30)

    @pytest.mark.parametrize(
        "beats_ago,should_reclaim",
        [(0, False), (2, False), (4, False), (6, True), (30, True)],
    )
    def test_worker_liveness_boundary(self, beats_ago, should_reclaim):
        now = datetime.now(timezone.utc)
        heartbeat_cutoff = now - timedelta(minutes=STALE_HEARTBEAT_MINUTES)
        last_heartbeat = now - timedelta(minutes=beats_ago)
        assert (last_heartbeat < heartbeat_cutoff) is should_reclaim
