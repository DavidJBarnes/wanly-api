"""Guard against per-segment fields reaching POST /segments but not job creation.

`faceswap_model` and `faceswap_pixel_boost` shipped wired into the append path only. A job
created with them in `first_segment` silently dropped both and fell back to daemon defaults --
invisible unless you query the row, because the defaults happened to be what was wanted.

Segment 0 is created by a different code path from every later segment, so anything added to
SegmentCreate has to be wired twice. This test fails when the two drift.
"""

import ast
import inspect
from pathlib import Path

from app.schemas.segments import SegmentCreate


def _kwargs_passed_to_segment(source_file: str) -> set[str]:
    """Every kwarg any Segment(...) call in the file passes."""
    tree = ast.parse(Path(source_file).read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "Segment"):
            names |= {kw.arg for kw in node.keywords if kw.arg}
    return names


# Fields that legitimately never come from the client on segment 0.
_NOT_CLIENT_SUPPLIED = {"index", "prompt", "duration_seconds", "speed", "loras"}


def test_job_creation_persists_the_same_faceswap_fields_as_the_append_path():
    from_jobs = _kwargs_passed_to_segment("app/routes/jobs.py")
    from_segments = _kwargs_passed_to_segment("app/routes/segments.py")
    faceswap_fields = {f for f in SegmentCreate.model_fields if f.startswith("faceswap_")}
    missing = (faceswap_fields & from_segments) - from_jobs
    assert not missing, f"first_segment silently drops: {sorted(missing)}"


def test_the_two_new_tunables_specifically_round_trip():
    """These are the ones that regressed; pin them by name so a rename cannot hide it."""
    from_jobs = _kwargs_passed_to_segment("app/routes/jobs.py")
    assert "faceswap_model" in from_jobs
    assert "faceswap_pixel_boost" in from_jobs
