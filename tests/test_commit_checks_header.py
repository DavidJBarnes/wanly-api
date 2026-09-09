"""artifact-commit believes a checkpoint only after reading its header (2026-09-08).

A zero-filled 654 MB file passed the size check, was recorded as Me_v2_final, and every
render with that character then failed inside ComfyUI with "Expecting value: line 1 column 1".
"""
import inspect
import json
import struct

from app import s3
from app.routes import training


class _Body:
    def __init__(self, b): self._b = b
    def read(self): return self._b


class _Client:
    def __init__(self, blob): self.blob = blob; self.ranges = []
    def get_object(self, Bucket, Key, Range):
        self.ranges.append(Range)
        a, b = Range.split("=")[1].split("-")
        return {"Body": _Body(self.blob[int(a):int(b) + 1])}


def _install(monkeypatch, blob):
    c = _Client(blob)
    monkeypatch.setattr(s3, "_client_for_bucket", lambda bucket: c)
    return c


def test_a_real_header_passes_with_two_small_reads(monkeypatch):
    header = json.dumps({"__metadata__": {}, "w": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    c = _install(monkeypatch, struct.pack("<Q", len(header)) + header + b"\0" * 4)
    assert s3.safetensors_header_ok("s3://ltx-loras/character/x.safetensors")
    assert c.ranges == ["bytes=0-7", f"bytes=8-{7 + len(header)}"]


def test_a_zero_filled_file_fails(monkeypatch):
    _install(monkeypatch, b"\0" * 4096)
    assert not s3.safetensors_header_ok("s3://ltx-loras/character/Me_v2_final.safetensors")


def test_a_truncated_header_fails(monkeypatch):
    _install(monkeypatch, struct.pack("<Q", 500) + b"{\"a\":")
    assert not s3.safetensors_header_ok("s3://ltx-loras/character/x.safetensors")


def test_commit_checks_the_header_after_presence_and_before_recording():
    src = inspect.getsource(training.commit_training_artifact)
    assert "s3.safetensors_header_ok, uri" in src
    # After presence and size (a truncated PUT keeps its 409), before anything is recorded.
    assert (src.index("head_object, uri") < src.index("MIN_ARTIFACT_BYTES")
            < src.index("safetensors_header_ok, uri") < src.index("_record_checkpoint(job, uri)"))
