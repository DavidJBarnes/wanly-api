"""wanly-api#434: measuring face size must never hold a whole set's bytes at once.

The Datasets page measures every card on sight. After the living-datasets backfill that was
~1,000 stills across a dozen cards at once, each request downloading ALL its images together;
the 2 GB API box thrashed into a hang. Now: one chunk in memory, one set at a time.
"""
import asyncio

import pytest

from app import face_size


@pytest.fixture
def fakes(monkeypatch):
    state = {"live": 0, "peak": 0, "measuring": 0, "peak_measuring": 0}

    def download(uri):
        state["live"] += 1
        state["peak"] = max(state["peak"], state["live"])
        return b"x" * 10

    async def measure(blobs):
        state["measuring"] += 1
        state["peak_measuring"] = max(state["peak_measuring"], state["measuring"])
        await asyncio.sleep(0.01)
        state["live"] -= len(blobs)
        state["measuring"] -= 1
        return [{"face_px": 100} for _ in blobs]

    monkeypatch.setattr(face_size.s3, "download_bytes", download)
    monkeypatch.setattr(face_size, "measure_blobs", measure)
    return state


@pytest.mark.asyncio
async def test_a_big_set_holds_one_chunk_at_a_time(fakes):
    uris = [f"s3://b/{i}.png" for i in range(166)]
    out = await face_size.measure_uris(uris)
    assert len(out) == 166
    assert fakes["peak"] <= face_size.MEASURE_CHUNK


@pytest.mark.asyncio
async def test_sets_asked_together_take_turns(fakes):
    sets = [[f"s3://b/{s}-{i}.png" for i in range(40)] for s in range(12)]
    outs = await asyncio.gather(*(face_size.measure_uris(u) for u in sets))
    assert all(len(o) == 40 for o in outs)
    assert fakes["peak_measuring"] == 1
    assert fakes["peak"] <= face_size.MEASURE_CHUNK


@pytest.mark.asyncio
async def test_an_undownloadable_image_is_left_out(monkeypatch, fakes):
    def download(uri):
        if uri.endswith("gone.png"):
            raise FileNotFoundError(uri)
        return b"x"
    monkeypatch.setattr(face_size.s3, "download_bytes", download)
    out = await face_size.measure_uris(["s3://b/a.png", "s3://b/gone.png"])
    assert set(out) == {"s3://b/a.png"}
