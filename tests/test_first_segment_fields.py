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
    """Every kwarg any Segment(...) call in the file passes, unioned."""
    names: set[str] = set()
    for _, kwargs in _segment_construction_sites(source_file):
        names |= kwargs
    return names


def _segment_construction_sites(source_file: str) -> list[tuple[str, set[str]]]:
    """(enclosing function, kwargs) for every Segment(...) construction in the file.

    PER SITE, not unioned over the file. Unioning is how the ltx_recipe bug got through: the
    re-roll builds a Segment in the same module as the append path, so `ltx_recipe` appearing
    in one made the whole file look correct while a re-rolled take silently lost its recipe
    and rendered free-form.
    """
    tree = ast.parse(Path(source_file).read_text())
    sites: list[tuple[str, set[str]]] = []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(fn):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "Segment"):
                sites.append((fn.name, {kw.arg for kw in node.keywords if kw.arg}))
    return sites


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


def test_job_creation_persists_ltx_recipe():
    """A recipe render creates its job and segment 0 in one call, so this is THE path it uses.

    Dropped here, the segment would generate with the daemon's defaults instead of the
    validated recipe -- and would look fine, because a plausible clip comes back either way.
    The graph hash recorded against it would then describe a configuration that never ran.
    """
    from_jobs = _kwargs_passed_to_segment("app/routes/jobs.py")
    assert "ltx_recipe" in from_jobs, "first_segment silently drops ltx_recipe"


def test_every_client_supplied_segment_field_is_wired_into_both_paths():
    """The general form of the two tests above.

    Both previous failures were the same shape -- a field added to the append path and not to
    job creation -- caught only after the fact, and only because someone queried the row. The
    faceswap guard was written for one family of fields and the ltx one for a single field;
    neither would catch the NEXT field added. This does.

    A field that legitimately never comes from the client on segment 0 goes in
    _NOT_CLIENT_SUPPLIED above, which makes the exemption a deliberate, reviewable line rather
    than a silent omission.
    """
    from_jobs = _kwargs_passed_to_segment("app/routes/jobs.py")
    from_segments = _kwargs_passed_to_segment("app/routes/segments.py")
    client_fields = set(SegmentCreate.model_fields) - _NOT_CLIENT_SUPPLIED
    missing = (client_fields & from_segments) - from_jobs
    assert not missing, (
        "fields wired into the append path but not job creation: "
        f"{sorted(missing)} -- segment 0 would silently drop them"
    )


# What a segment IS, as opposed to what happened to it. Every construction of a Segment has to
# carry these or the thing it builds is a different shot.
_SHOT_DEFINING = {"prompt", "ltx_recipe"}

# Sites that legitimately do not, each with its reason. Both are CARRIERS — a segment row used
# to hang reprocess work off a job, at a sentinel index far above any real segment. They render
# nothing, so there is no shot for a recipe to define.
#
# The list is the point: an exemption has to be argued for here rather than happening by
# omission, which is exactly how _roll_new_take lost ltx_recipe.
_EXEMPT: dict[str, set[str]] = {
    "make_hologram": {"ltx_recipe"},    # AR hologram carrier, index 1000
    "create_smashcut": {"ltx_recipe"},  # ffmpeg concat carrier, index 2000
}


def test_every_segment_construction_carries_what_defines_the_shot():
    """Checked PER SITE, because there are three and they drift independently.

    add_segment, job creation's segment 0, and _roll_new_take. The last one lost `ltx_recipe`
    for the whole time the column existed: a re-rolled LTX take came back with no character
    LoRA, no trigger and no per-stage strengths — a different shot, which is the exact
    opposite of what a re-roll is for. It went unnoticed because the previous version of this
    test unioned kwargs across a FILE, and _roll_new_take shares a module with add_segment.
    """
    sites = (
        _segment_construction_sites("app/routes/segments.py")
        + _segment_construction_sites("app/routes/jobs.py")
    )
    assert len(sites) >= 3, f"expected at least 3 construction sites, found {len(sites)}"
    for fn_name, kwargs in sites:
        missing = (_SHOT_DEFINING - kwargs) - _EXEMPT.get(fn_name, set())
        assert not missing, (
            f"{fn_name}() builds a Segment without {sorted(missing)} — the take it creates "
            f"is a different shot from the one it is meant to reproduce"
        )
