"""Reclaiming a claim whose RESPONSE was lost (wanly-api#242).

GET /segments/next mutates: the segment is marked and assigned before the answer is sent,
so a transport failure on the way back leaves it owned by a worker that never learned of it.
That worker stays healthy and heartbeating, so the dead-worker rules never fire and the
segment sits forever with the queue stopped beside an idle GPU.

The danger in fixing this is the opposite failure, which has already happened once: on
2026-08-15 an age rule that applied to a LIVE worker stole renders mid-flight and two GPUs
converged on the same segment. Every test here is about not doing that again.
"""
import inspect

from app.routes import segments as mod


def test_the_rule_requires_all_three_conditions():
    """Idle status alone is not enough, and neither is age.

    The daemon's status push can fail, and the heartbeat only re-pushes on a CHANGE — so a
    worker can be rendering while marked idle. The empty progress log is the evidence that
    cannot lie, and it must be part of the conjunction.
    """
    src = inspect.getsource(mod.claim_next_segment)
    assert 'Worker.status == "online-idle"' in src
    assert "Segment.claimed_at < orphan_cutoff" in src
    assert "Segment.progress_log.is_(None)" in src


def test_the_grace_period_is_far_longer_than_the_first_progress_line():
    """The daemon logs "[1/6] Downloading start image..." within seconds of claiming.

    The window only has to outlast that round trip. Too short and a slow first write looks
    like an orphan; too long and the queue stalls for no reason.
    """
    assert mod.ORPHANED_CLAIM_MINUTES >= 2
    assert mod.ORPHANED_CLAIM_MINUTES < mod.STALE_HEARTBEAT_MINUTES


def test_a_busy_worker_is_never_a_candidate():
    """The 2026-08-15 incident in one assertion.

    A worker that received its claim reports online-busy BEFORE it starts rendering, so it
    can never match. If this ever widens to include busy workers, long renders get stolen
    mid-flight and two GPUs converge on one segment.
    """
    src = inspect.getsource(mod.claim_next_segment)
    assert 'Worker.status == "online-idle"' in src
    assert 'Worker.status != "online-busy"' not in src, (
        "must match idle explicitly, not exclude busy — draining and offline workers "
        "would otherwise become candidates too"
    )


def test_a_segment_with_progress_is_never_reclaimed_by_this_rule():
    """Progress is proof the worker got the segment and is working on it.

    Reclaiming one that has written progress is the mid-flight theft this must not do.
    """
    src = inspect.getsource(mod.claim_next_segment)
    # Anchor on the condition itself, not on a variable name that also appears where it is
    # declared — the three checks must live in ONE and_(), or they are not a conjunction.
    i = src.index('Worker.status == "online-idle"')
    clause = src[i:i + 900]   # the rule carries explanatory comments between conditions
    assert "progress_log" in clause, "the progress check must sit in the same and_() as the idle check"
    assert "orphan_cutoff" in clause, "the age check must sit in the same and_() too"


def test_the_log_says_which_rule_fired():
    """"The worker died" and "the worker is fine but never got the answer" have completely
    different fixes. Collapsing them into one message costs the next diagnosis."""
    src = inspect.getsource(mod.claim_next_segment)
    assert "claim response lost" in src
    assert "worker heartbeat stale" in src
    assert "worker row gone" in src
