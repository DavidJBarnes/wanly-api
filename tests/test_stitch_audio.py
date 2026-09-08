"""Audio through the stitch (console#475).

LTX renders carry AAC 48 kHz stereo, and the rule is deliberately binary: every clip
audible -> the final keeps it with sound; any clip silent -> concat entries get padded
with synthesized silence so the `-c copy` list stays uniform. The tests run REAL ffmpeg
against synthesized clips — the failure modes here were all command-level (a silent black
clip inside a copy list, an `-an` crossfade), and argv-shape assertions would have let a
filter-graph typo pass silently.
"""

import json
import shutil
import subprocess

import pytest

from app.stitch import (
    AUDIO_SAMPLE_RATE,
    _crossfade_concat,
    _generate_black,
    _has_audio_stream,
    _pad_silent_track,
)
from app import stitch as stitch_module

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")

NEEDS_FFMPEG = pytest.mark.skipif(not FFMPEG or not FFPROBE, reason="no ffmpeg/ffprobe")


def _encoders() -> str:
    if not FFMPEG:
        return ""
    try:
        return subprocess.run([FFMPEG, "-hide_banner", "-encoders"],
                              capture_output=True, text=True).stdout
    except Exception:
        return ""


HAS_LIBX264 = "libx264" in _encoders()


@pytest.fixture
def any_encoder(request, monkeypatch):
    """Run _crossfade_concat/_generate_black regardless of libx264 availability.

    app.stitch hardcodes libx264; CI runners and the deployed container have it, a stock
    workstation ffmpeg may not. When it is missing, argv gets libx264 swapped for mpeg4
    and the test runs anyway — the premise under test is the filter graph, the maps and
    the stream layout, and per-version encoder substitution states itself instead of
    pretending. A no-op where libx264 exists.
    """
    if HAS_LIBX264:
        yield
        return
    real_run = subprocess.run

    def swapped(cmd, **kwargs):
        if isinstance(cmd, list) and "libx264" in cmd:
            cmd = ["mpeg4" if c == "libx264" else c for c in cmd]
        return real_run(cmd, **kwargs)

    monkeypatch.setattr(stitch_module.subprocess, "run", swapped)
    yield


def _stream_probe(path: str) -> list[dict]:
    """Every stream {codec_type, codec_name, sample_rate} of an output file."""
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "stream=codec_type,codec_name,sample_rate", "-of", "json", path],
        capture_output=True, text=True, check=True,
    )
    return json.loads(proc.stdout)["streams"]


def _audio_streams(path: str) -> list[dict]:
    return [s for s in _stream_probe(path) if s["codec_type"] == "audio"]


def _make_clip(path: str, audible: bool) -> None:
    """A 1-second test clip: testsrc video, sine+48k stereo audio if audible."""
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "lavfi", "-i", "testsrc=s=64x64:d=1",
    ]
    if audible:
        cmd += ["-f", "lavfi", "-i", "sine=f=440:d=1"]
    cmd += ["-c:v", "mpeg4"]
    if audible:
        cmd += ["-c:a", "aac", "-ar", str(AUDIO_SAMPLE_RATE), "-ac", "2"]
    subprocess.run(cmd + ["-shortest", path], check=True)


class TestProbeHasAudioStream:
    @NEEDS_FFMPEG
    def test_audible_yes_silent_no(self, tmp_path):
        audible = str(tmp_path / "a.mp4")
        _make_clip(audible, audible=True)
        assert _has_audio_stream(audible) is True

        silent = str(tmp_path / "s.mp4")
        _make_clip(silent, audible=False)
        assert _has_audio_stream(silent) is False


class TestPadSilentTrack:
    @NEEDS_FFMPEG
    def test_a_padded_clip_matches_the_stitch_layout(self, tmp_path):
        """Either every entry of the copy list has audio or none does. Padding synthesizes
        silence AT THE AGREED LAYOUT — a silent AAC whose rate differs from the audible
        neighbors would make the copy list uniform in stream PRESENCE only, which is not
        uniform enough for `-c copy`."""
        silent = str(tmp_path / "s.mp4")
        _make_clip(silent, audible=False)
        padded = str(tmp_path / "p.mp4")
        _pad_silent_track(silent, padded)

        audio = _audio_streams(padded)
        assert audio and audio[0]["codec_name"] == "aac"
        assert audio[0]["sample_rate"] == str(AUDIO_SAMPLE_RATE)

    @NEEDS_FFMPEG
    def test_the_padded_video_is_untouched(self, tmp_path):
        """-c:v copy keeps the padding from re-encoding picture: cheap, and no quality loss."""
        silent = str(tmp_path / "s.mp4")
        _make_clip(silent, audible=False)
        padded = str(tmp_path / "p.mp4")
        _pad_silent_track(silent, padded)
        assert [s["codec_name"] for s in _stream_probe(padded) if s["codec_type"] == "video"] == ["mpeg4"]


class TestUniformConcatCopy:
    @NEEDS_FFMPEG
    def test_an_audio_uniform_list_copies_with_sound(self, tmp_path):
        """The shape a modern all-LTX job stitches in: every entry has AAC 48k stereo, so
        `-c copy` carries it — no re-encode, no `-an`, no silent出品 surprise."""
        a, p, b = str(tmp_path / "a.mp4"), str(tmp_path / "p.mp4"), str(tmp_path / "b.mp4")
        _make_clip(a, audible=True)
        _pad_silent_track(a, p)
        _make_clip(b, audible=True)

        lst = tmp_path / "concat.txt"
        lst.write_text(f"file '{a}'\nfile '{p}'\nfile '{b}'\n")
        out = str(tmp_path / "final.mp4")
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0",
             "-i", str(lst), "-c", "copy", out],
            check=True,
        )

        audio = _audio_streams(out)
        assert audio and audio[0]["codec_name"] == "aac"
        assert audio[0]["sample_rate"] == str(AUDIO_SAMPLE_RATE)
        # 3 x 1s in; nothing re-encoded, nothing lost.
        assert 2.9 < _duration(out) < 3.15


def _duration(path: str) -> float:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True, check=True,
    )
    return float(proc.stdout.strip())


@pytest.mark.skipif(not HAS_LIBX264, reason="the black clip really re-encodes; no libx264 here")
class TestBlackClip:
    @NEEDS_FFMPEG
    def test_the_black_clip_carries_silence_at_the_stitch_layout(self, tmp_path):
        """A black insert sits in the copy list BETWEEN audible segments; pre-#475 it was
        video-only with `-an`, and a copy list requires uniform streams. This is the test
        that would have caught that — assert the stream is present and shaped to match."""
        out = str(tmp_path / "black.mp4")
        _generate_black(out, 1.0, 64, 64, 25)

        audio = _audio_streams(out)
        assert audio and audio[0]["codec_name"] == "aac"
        assert audio[0]["sample_rate"] == str(AUDIO_SAMPLE_RATE)


@pytest.mark.usefixtures("any_encoder")
class TestCrossfadeAudio:
    @NEEDS_FFMPEG
    def test_carry_audio_fades_the_sound_through_the_seams(self, tmp_path):
        """carry_audio=True (every clip audible): audio is chained per-boundary with the
        same overlap arithmetic as the video, so picture and sound shorten together."""
        a, b = str(tmp_path / "a.mp4"), str(tmp_path / "b.mp4")
        _make_clip(a, audible=True)
        _make_clip(b, audible=True)

        out = str(tmp_path / "x.mp4")
        total = _crossfade_concat([a, b], [1.0, 1.0], 0.5, 25, out, carry_audio=True)

        assert abs(total - 1.5) < 0.05
        audio = _audio_streams(out)
        assert audio and audio[0]["codec_name"] == "aac"
        assert abs(_duration(out) - 1.5) < 0.1

    @NEEDS_FFMPEG
    def test_without_audio_the_output_is_honestly_silent(self, tmp_path):
        """carry_audio=False is the fallback when any clip lacks a stream: video-only, by
        decision, not by accident. An artifact of that shape declares it, instead of the
        old `-an` silent behavior with no flag on the record."""
        a, b = str(tmp_path / "a.mp4"), str(tmp_path / "b.mp4")
        _make_clip(a, audible=True)
        _make_clip(b, audible=True)

        out = str(tmp_path / "x.mp4")
        _crossfade_concat([a, b], [1.0, 1.0], 0.5, 25, out, carry_audio=False)
        assert _audio_streams(out) == []
