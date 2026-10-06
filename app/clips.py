"""Video clips in a dataset (wanly-api#411): identity in motion for LTX character LoRAs.

A clip is a dataset item whose URI ends in `.mp4`. It sits in `Dataset.images` beside the
stills and is keyed by URI like everything else -- captions, scores, the training snapshot --
so nothing that stores or moves dataset items needed a second list or a column.

NORMALIZED ON UPLOAD, ONCE. A phone clip arrives at 30 or 60 fps, 1080p or 4K, with an audio
track, sometimes as .mov. What the trainer wants is fixed (wanly-gpu-docker#189), so it is
produced here and the stored object is already it:

    25 fps          LTX-2's own rate. musubi resamples anything else to 25 by dropping
                    frames, which is a worse job than ffmpeg's and happens on every cache.
    no audio        the recipe trains no audio weights (video_sa_ca_ff), so a track is bytes
                    on every download and nothing else.
    long edge 768   above the clip group's 512 bucket ceiling with room to crop, and a 4K
                    clip is 50x the bytes for nothing.
    <= 10 s         longer is longer to download, cache and score, for windows the trainer
                    spreads across the first ten seconds anyway.
    H.264 CRF 18    visually lossless; the source is already lossy. Preset veryfast: this
                    runs inside the upload request on the API's small EC2 box, and a slower
                    preset buys a smaller file nobody needs at the cost of a batch timing out.

A CLIP UNDER 51 FRAMES IS REFUSED. The trainer cuts three 49-frame windows (8n+1, LTX's
latent frame count) spread across it; a shorter clip yields fewer distinct windows than both
sides count, or none at all -- it would sit in the set, count as an item, and never train.
"""
from __future__ import annotations

import asyncio
import json
import statistics
import subprocess
import tempfile
from pathlib import Path

#: What an upload may be. Everything is stored as .mp4.
CLIP_SUFFIXES = {".mp4", ".mov", ".m4v", ".webm"}
FPS = 25
MAX_SECONDS = 10.0
#: The trainer's window length (wanly-gpu-docker#189's target_frames).
WINDOW_FRAMES = 49
#: The shortest clip kept: long enough for the trainer's 3 windows to land on DIFFERENT
#: frames. musubi places them at whole-frame offsets across (frames - 49); at 49 frames all
#: three are the same window and the API and trainer would count three samples where one
#: trains. Kept in step with training_plan.CLIP_WINDOWS by a test.
MIN_FRAMES = WINDOW_FRAMES + 2
LONG_EDGE = 768
#: Frames a clip is scored on, spread evenly. Five catches a different person walking in
#: halfway without making a 30-clip set a 150-image embed call.
SCORE_FRAMES = 5
#: Frames on the caption's contact sheet, in reading order (2x2).
CAPTION_FRAMES = 4


class ClipError(ValueError):
    """A clip that cannot become a training clip. The message is shown to the user."""


def is_clip(uri: str) -> bool:
    return uri.lower().endswith(".mp4")


def _run(argv: list[str], timeout: int = 300) -> subprocess.CompletedProcess:
    proc = subprocess.run(argv, capture_output=True, timeout=timeout)
    if proc.returncode != 0:
        raise ClipError(f"{argv[0]} failed: {proc.stderr.decode(errors='replace')[-400:]}")
    return proc


def _probe(path: Path) -> tuple[float, int]:
    """(duration seconds, decoded frame count) of the first video stream."""
    out = _run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
                "-show_entries", "stream=nb_read_frames:format=duration", "-of", "json",
                str(path)], timeout=120)
    d = json.loads(out.stdout or b"{}")
    streams = d.get("streams") or []
    if not streams:
        raise ClipError("it has no video stream")
    frames = int(streams[0].get("nb_read_frames") or 0)
    duration = float((d.get("format") or {}).get("duration") or 0.0)
    return duration, frames


def _normalize_sync(data: bytes, suffix: str) -> bytes:
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / f"in{suffix}"
        dst = Path(tmp) / "out.mp4"
        src.write_bytes(data)
        _probe(src)  # a file ffprobe cannot read is refused before ffmpeg says something worse
        # Long edge to 768 (never up), the short edge following with an even size -- H.264
        # with yuv420p refuses odd dimensions.
        scale = (f"scale='if(gte(iw,ih),min({LONG_EDGE},iw),-2)':"
                 f"'if(gte(iw,ih),-2,min({LONG_EDGE},ih))',"
                 f"scale=trunc(iw/2)*2:trunc(ih/2)*2")
        _run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-t", str(MAX_SECONDS),
              "-an", "-vf", f"fps={FPS},{scale}", "-c:v", "libx264", "-crf", "18",
              "-preset", "veryfast", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
              str(dst)])
        _, frames = _probe(dst)
        if frames < MIN_FRAMES:
            raise ClipError(
                f"it is {frames / FPS:.1f} s long — a clip needs at least "
                f"{MIN_FRAMES / FPS:.1f} s ({MIN_FRAMES} frames at {FPS} fps) to train on")
        return dst.read_bytes()


async def normalize(data: bytes, suffix: str) -> bytes:
    """The stored form of an uploaded clip (see the module docstring). Raises ClipError."""
    try:
        return await asyncio.to_thread(_normalize_sync, data, suffix.lower())
    except subprocess.TimeoutExpired as e:
        raise ClipError("it took too long to convert") from e


def _times(duration: float, n: int) -> list[float]:
    """n timestamps at the middles of n equal slices: never the first or last frame, which
    on a phone clip are the likeliest to be a fade, a blur or a thumb."""
    return [duration * (i + 0.5) / n for i in range(n)]


def _frames_sync(data: bytes, n: int) -> list[bytes]:
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "clip.mp4"
        src.write_bytes(data)
        duration, _ = _probe(src)
        out = []
        for i, t in enumerate(_times(duration, n)):
            jpg = Path(tmp) / f"f{i}.jpg"
            _run(["ffmpeg", "-y", "-v", "error", "-ss", f"{t:.3f}", "-i", str(src),
                  "-frames:v", "1", "-q:v", "2", str(jpg)], timeout=60)
            out.append(jpg.read_bytes())
        return out


async def frames(data: bytes, n: int = SCORE_FRAMES) -> list[bytes]:
    """n evenly spaced stills from a clip, as JPEG bytes."""
    return await asyncio.to_thread(_frames_sync, data, n)


def _sheet_sync(data: bytes) -> bytes:
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "clip.mp4"
        src.write_bytes(data)
        duration, _ = _probe(src)
        parts = []
        for i, t in enumerate(_times(duration, CAPTION_FRAMES)):
            jpg = Path(tmp) / f"f{i}.jpg"
            _run(["ffmpeg", "-y", "-v", "error", "-ss", f"{t:.3f}", "-i", str(src),
                  "-frames:v", "1", "-vf", "scale=512:-2", "-q:v", "3", str(jpg)], timeout=60)
            parts.append(jpg)
        sheet = Path(tmp) / "sheet.jpg"
        inputs = [a for p in parts for a in ("-i", str(p))]
        _run(["ffmpeg", "-y", "-v", "error", *inputs, "-filter_complex",
              "[0][1]hstack[top];[2][3]hstack[bot];[top][bot]vstack", "-q:v", "3",
              str(sheet)], timeout=60)
        return sheet.read_bytes()


async def contact_sheet(data: bytes) -> bytes:
    """Four frames of a clip in a 2x2 grid, in reading order -- ONE image, because the
    captioner takes one, and a vision model reads a grid of moments as a sequence well enough
    to say what changes between them."""
    return await asyncio.to_thread(_sheet_sync, data)


def clip_score(cosines: list[float | None]) -> float | None:
    """A clip's likeness to the anchor: the MEDIAN over the frames that have a face, or None
    when none does.

    The median, not the minimum: identity in motion means head turns, and a frame caught
    mid-turn scores low against a frontal anchor without being somebody else. Not the mean
    either, for the same reason in the other direction -- one profile frame drags a mean
    under the floor. A clip that is MOSTLY somebody else still lands low.
    """
    have = [c for c in cosines if c is not None]
    return round(statistics.median(have), 4) if have else None
