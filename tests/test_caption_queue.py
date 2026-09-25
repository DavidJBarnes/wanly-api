"""Describes take a turn instead of racing the captioner.

ollama runs OLLAMA_NUM_PARALLEL=1, so concurrent describes were ALWAYS serialised -- they
just did their waiting at the captioner, inside an HTTP request that could time out.
Measured on the 3090 at ~25s a call and two calls an image, latency climbed 18s, 28s, 27s,
42s, 57s, 1m10, 1m27 and then requests began failing -- with the scene half of the same
image already stored and the motion half lost, which is what "scene described, but the
motion caption failed" actually was.

Waiting here instead buys two things the race could not: a known order, and a position every
response can report.
"""
import asyncio

import pytest

from app.caption_queue import CaptionQueue


@pytest.mark.asyncio
async def test_only_one_caption_runs_at_a_time():
    """The property everything else rests on. A second body entered while the first is
    inside its block would mean the captioner is being raced again."""
    q = CaptionQueue()
    inside = 0
    peak = 0

    async def one(path):
        nonlocal inside, peak
        async with q.turn(path):
            inside += 1
            peak = max(peak, inside)
            await asyncio.sleep(0.02)
            inside -= 1

    await asyncio.gather(*(one(f"p{i}") for i in range(8)))
    assert peak == 1, f"{peak} captions ran at once"


@pytest.mark.asyncio
async def test_turns_are_taken_in_arrival_order():
    """Out of order is not merely untidy: the position reported to the UI would be a lie,
    and the image you are watching could be overtaken by one queued after it."""
    q = CaptionQueue()
    order = []

    async def one(path):
        async with q.turn(path):
            order.append(path)
            await asyncio.sleep(0.01)

    tasks = []
    for i in range(6):
        tasks.append(asyncio.create_task(one(f"p{i}")))
        await asyncio.sleep(0.001)      # so arrival order is unambiguous
    await asyncio.gather(*tasks)
    assert order == [f"p{i}" for i in range(6)]


@pytest.mark.asyncio
async def test_position_counts_what_is_ahead():
    q = CaptionQueue()
    started = asyncio.Event()
    release = asyncio.Event()

    async def first():
        async with q.turn("first"):
            started.set()
            await release.wait()

    async def waiter(path):
        async with q.turn(path):
            pass

    t1 = asyncio.create_task(first())
    await started.wait()
    t2 = asyncio.create_task(waiter("second"))
    t3 = asyncio.create_task(waiter("third"))
    await asyncio.sleep(0.01)

    assert q.status("first") == {"status": "running", "position": 0, "depth": 3}
    assert q.status("second")["position"] == 1
    assert q.status("third")["position"] == 2

    release.set()
    await asyncio.gather(t1, t2, t3)


@pytest.mark.asyncio
async def test_a_path_that_was_never_queued_reads_as_nothing():
    """Null rather than 0, because 0 is a real position -- it means "running now"."""
    q = CaptionQueue()
    assert q.status("never")["status"] is None
    assert q.status("never")["position"] is None
    assert q.depth() == 0


@pytest.mark.asyncio
async def test_depth_counts_the_one_in_progress_too():
    """"2 waiting" while a third is mid-caption understates the wait by a whole caption."""
    q = CaptionQueue()
    release = asyncio.Event()
    started = asyncio.Event()

    async def hold():
        async with q.turn("running"):
            started.set()
            await release.wait()

    t = asyncio.create_task(hold())
    await started.wait()
    assert q.depth() == 1
    release.set()
    await t
    assert q.depth() == 0


@pytest.mark.asyncio
async def test_a_failure_does_not_wedge_the_queue():
    """One unreadable image must not hold up everything behind it -- the turn is released
    on the way out however the body ends."""
    q = CaptionQueue()

    async def boom():
        async with q.turn("bad"):
            raise RuntimeError("could not read the image")

    with pytest.raises(RuntimeError):
        await boom()

    assert q.depth() == 0
    async with q.turn("good"):
        pass
    assert q.depth() == 0


@pytest.mark.asyncio
async def test_a_cancelled_waiter_leaves_no_phantom_in_the_line():
    """A client that gives up must not inflate every position reported after it."""
    q = CaptionQueue()
    started = asyncio.Event()
    release = asyncio.Event()

    async def hold():
        async with q.turn("holder"):
            started.set()
            await release.wait()

    async def giver_upper():
        async with q.turn("gone"):
            pass

    t1 = asyncio.create_task(hold())
    await started.wait()
    t2 = asyncio.create_task(giver_upper())
    await asyncio.sleep(0.01)
    assert q.depth() == 2

    t2.cancel()
    with pytest.raises(asyncio.CancelledError):
        await t2
    assert q.status("gone")["status"] is None
    assert q.depth() == 1, "the abandoned waiter is still being counted"

    release.set()
    await t1
