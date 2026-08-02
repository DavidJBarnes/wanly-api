"""Tests for the per-segment seed re-anchor flag.

Context: seed_faceswap started as a GLOBAL app setting gated on "a later segment already
exists at claim time". That gate can never pass — a job is created with segment 0 only and
continuations are appended after it runs, so at claim time there is never a successor. The
feature was dead in every job it ever ran against (26 multi-segment jobs, 0 activations),
and it had no tests, which is why it stayed dead.

These tests pin the three things that made it broken:
  1. no successor gate on the claim
  2. the flag lives on the segment and survives create -> persist -> claim
  3. the global app setting is gone, so it cannot be resurrected as a second source of truth

Unit-level (no live DB), matching the rest of this suite. The AST checks follow the
hand-built-response guard in test_lynx_engine.py: a field can exist on the schema and the
model and still be silently dropped by a field-by-field construction.
"""

import ast
from pathlib import Path

import pytest

from app.models import Segment
from app.schemas.app_settings import AppSettingsResponse, AppSettingsUpdate
from app.schemas.segments import SegmentClaimResponse, SegmentCreate, SegmentResponse

ROOT = Path(__file__).resolve().parents[1]
SEGMENTS_ROUTE = ROOT / "app" / "routes" / "segments.py"
JOBS_ROUTE = ROOT / "app" / "routes" / "jobs.py"


class TestSchemaContract:
    def test_segment_create_defaults_off(self):
        assert SegmentCreate(prompt="x").seed_faceswap is False

    def test_segment_create_accepts_the_flag(self):
        assert SegmentCreate(prompt="x", seed_faceswap=True).seed_faceswap is True

    def test_claim_response_carries_the_flag(self):
        """The daemon reads this field; if it is missing the re-anchor silently never runs."""
        assert "seed_faceswap" in SegmentClaimResponse.model_fields

    def test_segment_response_carries_the_flag(self):
        """Console needs it back to render the checkbox state on an existing segment."""
        assert "seed_faceswap" in SegmentResponse.model_fields

    def test_model_has_the_column(self):
        assert hasattr(Segment, "seed_faceswap")

    def test_seed_only_is_valid_without_whole_video_faceswap(self):
        """The re-anchor touches ONE frame; requiring the whole clip to be swapped as well
        would be a different (and much more expensive) feature. The daemon resolves the face
        from faceswap_image even when faceswap_enabled is False."""
        seg = SegmentCreate(
            prompt="x",
            seed_faceswap=True,
            faceswap_enabled=False,
            faceswap_image="s3://wanly-loras/faces/kelly.png",
        )
        assert seg.seed_faceswap is True
        assert seg.faceswap_enabled is False
        assert seg.faceswap_image == "s3://wanly-loras/faces/kelly.png"

    def test_whole_video_swap_without_seed_reanchor_is_valid(self):
        """The other direction: swapping the clip does not imply re-anchoring the seed."""
        seg = SegmentCreate(prompt="x", faceswap_enabled=True, seed_faceswap=False)
        assert seg.faceswap_enabled is True and seg.seed_faceswap is False


class TestGlobalSettingIsGone:
    """One source of truth. A lingering global would silently fight the per-segment value."""

    def test_not_on_the_settings_response(self):
        assert "seed_faceswap" not in AppSettingsResponse.model_fields

    def test_not_on_the_settings_update(self):
        assert "seed_faceswap" not in AppSettingsUpdate.model_fields

    def test_not_in_the_settings_defaults(self):
        src = (ROOT / "app" / "routes" / "app_settings.py").read_text()
        assert "seed_faceswap" not in src


class TestNoSuccessorGate:
    """THE regression. The claim must not condition the flag on a later segment existing."""

    def _claim_fn(self) -> ast.AST:
        tree = ast.parse(SEGMENTS_ROUTE.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "get_next_segment":
                return node
        # name may differ; fall back to the function containing the claim response
        for node in ast.walk(tree):
            if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name) \
                            and sub.func.id == "SegmentClaimResponse":
                        return node
        pytest.fail("could not locate the claim route function")

    def test_claim_does_not_query_for_a_successor(self):
        src = ast.unparse(self._claim_fn())
        assert "successor" not in src, (
            "the claim route still gates seed_faceswap on a successor segment; that gate "
            "can never pass because continuations are appended after segment 0 runs"
        )

    def test_claim_does_not_read_the_global_app_setting(self):
        src = ast.unparse(self._claim_fn())
        assert 'AppSetting, "seed_faceswap"' not in src
        assert "seed_faceswap_on" not in src

    def test_claim_resolves_the_flag_from_the_segment(self):
        src = ast.unparse(self._claim_fn())
        assert "segment.seed_faceswap" in src


class TestPersistence:
    """A field can exist everywhere and still be dropped by a field-by-field construction."""

    def _kwargs_for(self, path: Path, callee: str) -> list[set[str]]:
        tree = ast.parse(path.read_text())
        out = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                    and node.func.id == callee:
                out.append({kw.arg for kw in node.keywords if kw.arg})
        return out

    def test_add_segment_persists_the_flag(self):
        calls = self._kwargs_for(SEGMENTS_ROUTE, "Segment")
        assert calls, "no Segment(...) construction in app/routes/segments.py"
        assert any("seed_faceswap" in kw for kw in calls), (
            "POST /jobs/{id}/segments drops seed_faceswap, so continuation segments "
            "could never enable the re-anchor"
        )

    def test_job_creation_persists_the_flag_on_segment_zero(self):
        calls = self._kwargs_for(JOBS_ROUTE, "Segment")
        assert calls, "no Segment(...) construction in app/routes/jobs.py"
        assert all("seed_faceswap" in kw for kw in calls)

    def test_claim_response_passes_the_flag(self):
        calls = self._kwargs_for(SEGMENTS_ROUTE, "SegmentClaimResponse")
        assert len(calls) == 1, "expected exactly one claim construction site"
        assert "seed_faceswap" in calls[0]


class TestMigrationChain:
    """A duplicate revision id gives alembic two heads and `upgrade head` fails, which
    silently blocks every deploy. Both API deploys on 2026-08-02 died this way: this
    migration was numbered 050, which 050_add_lynx_engine.py already used."""

    def _chain(self):
        import re, glob, os
        nodes = {}
        for f in sorted(glob.glob(str(ROOT / "alembic" / "versions" / "*.py"))):
            src = open(f).read()
            rev = re.search(r'^revision(?::[^=]+)? *= *["\'](.+?)["\']', src, re.M)
            down = re.search(r'^down_revision(?::[^=]+)? *= *(?:["\'](.+?)["\']|None)', src, re.M)
            if rev:
                nodes.setdefault(rev.group(1), []).append(
                    (down.group(1) if down and down.lastindex else None, os.path.basename(f))
                )
        return nodes

    def test_no_duplicate_revision_ids(self):
        dupes = {k: [f for _, f in v] for k, v in self._chain().items() if len(v) > 1}
        assert not dupes, f"duplicate alembic revision ids will break `upgrade head`: {dupes}"

    def test_exactly_one_head(self):
        nodes = self._chain()
        downs = {d for v in nodes.values() for d, _ in v if d}
        heads = sorted(set(nodes) - downs)
        assert len(heads) == 1, f"alembic must have exactly one head, found {heads}"
