"""Tests for the Lynx identity-preserving engine's API surface.

These are unit-level (no live DB), matching the rest of this suite. The most valuable
case here is the hand-built-response guard: JobResponse/JobDetailResponse are constructed
field-by-field in the route handlers, so a field can exist on the schema and the model and
still be silently dropped on the wire.
"""

import ast
from pathlib import Path

import pytest

from app.models import Job, Segment
from app.schemas.jobs import JobCreate, JobDetailResponse, JobResponse
from app.schemas.segments import SegmentClaimResponse, SegmentResponse, SegmentStatusUpdate

# Every job-level Lynx tunable. Kept here as the single list the model, the schemas,
# the migration and the route constructions are all checked against.
LYNX_JOB_FIELDS = [
    "generation_engine",
    "lynx_subject_image",
    "lynx_ip_scale",
    "lynx_ref_scale",
    "lynx_cfg_scale",
    "lynx_start_percent",
    "lynx_end_percent",
    "lynx_ref_blocks_to_use",
    "lynx_ip_layers",
    "lynx_resampler",
    "lynx_steps",
    "lynx_cfg",
    "lynx_shift",
    "lynx_scheduler",
    "lynx_distill_strength",
]

ROOT = Path(__file__).resolve().parent.parent


class TestJobModel:
    @pytest.mark.parametrize("field", LYNX_JOB_FIELDS)
    def test_column_exists_and_is_nullable(self, field):
        """Nullable is the contract: NULL means 'use the daemon's settings default'."""
        column = Job.__table__.columns[field]
        assert column.nullable is True

    def test_segment_carries_identity_scores(self):
        assert Segment.__table__.columns["lynx_identity_scores"].nullable is True


class TestJobCreateSchema:
    def test_lynx_fields_default_to_none(self):
        body = JobCreate(
            name="j", width=832, height=480, fps=15,
            first_segment={"prompt": "p", "duration_seconds": 5.4},
        )
        for field in LYNX_JOB_FIELDS:
            assert getattr(body, field) is None, f"{field} should default to None"

    def test_accepts_a_full_lynx_job(self):
        body = JobCreate(
            name="lynx job", width=832, height=480, fps=15,
            generation_engine="lynx",
            lynx_subject_image="s3://bucket/face.png",
            lynx_ip_scale=0.7, lynx_ref_scale=0.6, lynx_cfg_scale=2.0,
            lynx_start_percent=0.0, lynx_end_percent=1.0,
            lynx_ref_blocks_to_use="0-20, 25",
            lynx_ip_layers="lite_ip.safetensors", lynx_resampler="lite_res.safetensors",
            lynx_steps=6, lynx_cfg=1.0, lynx_shift=8.0, lynx_scheduler="lcm",
            lynx_distill_strength=1.0,
            first_segment={"prompt": "p", "duration_seconds": 5.4},
        )
        assert body.generation_engine == "lynx"
        assert body.lynx_ip_scale == 0.7
        assert body.lynx_ref_scale == 0.6

    def test_zero_scale_survives_validation(self):
        """0.0 disables an adapter and must not be coerced to None/default."""
        body = JobCreate(
            name="j", width=832, height=480, fps=15, lynx_ip_scale=0.0,
            first_segment={"prompt": "p", "duration_seconds": 5.4},
        )
        assert body.lynx_ip_scale == 0.0


class TestResponseSchemas:
    @pytest.mark.parametrize("field", LYNX_JOB_FIELDS)
    def test_job_response_declares_field(self, field):
        assert field in JobResponse.model_fields

    @pytest.mark.parametrize("field", LYNX_JOB_FIELDS)
    def test_job_detail_response_inherits_field(self, field):
        assert field in JobDetailResponse.model_fields

    @pytest.mark.parametrize("field", LYNX_JOB_FIELDS)
    def test_claim_response_declares_field(self, field):
        """The daemon builds the graph from these, so the claim must carry them."""
        assert field in SegmentClaimResponse.model_fields

    def test_identity_scores_are_a_result_not_a_claim_input(self):
        # Written by the daemon on completion, read back on the segment...
        assert "lynx_identity_scores" in SegmentStatusUpdate.model_fields
        assert "lynx_identity_scores" in SegmentResponse.model_fields
        # ...but they cannot exist at claim time.
        assert "lynx_identity_scores" not in SegmentClaimResponse.model_fields

    def test_status_update_accepts_scores_payload(self):
        payload = {
            "scores": [0.61, 0.58], "mean": 0.595, "min": 0.58, "max": 0.61,
            "frames_sampled": 5, "frames_with_face": 2,
        }
        body = SegmentStatusUpdate(status="completed", lynx_identity_scores=payload)
        assert body.lynx_identity_scores == payload

    def test_status_update_scores_default_none(self):
        assert SegmentStatusUpdate(status="completed").lynx_identity_scores is None


class TestHandBuiltResponses:
    """JobResponse/JobDetailResponse are built field-by-field in app/routes/jobs.py.

    Pydantic silently drops any field the construction omits, even though it is declared
    on the schema — so a passing schema test alone would not catch the regression. This
    parses the route module and asserts every Lynx field is passed at every call site.
    """

    @staticmethod
    def _response_calls():
        tree = ast.parse((ROOT / "app" / "routes" / "jobs.py").read_text())
        calls = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                    and node.func.id in {"JobResponse", "JobDetailResponse"}:
                calls.append({kw.arg for kw in node.keywords if kw.arg})
        return calls

    def test_all_construction_sites_found(self):
        # 1 list response + 2 detail responses. If this changes, the parametrised test
        # below is silently covering fewer sites than it should.
        assert len(self._response_calls()) == 3

    @pytest.mark.parametrize("field", LYNX_JOB_FIELDS)
    def test_every_site_passes_every_lynx_field(self, field):
        for i, kwargs in enumerate(self._response_calls()):
            assert field in kwargs, (
                f"JobResponse/JobDetailResponse construction #{i} in app/routes/jobs.py "
                f"omits {field!r}; Pydantic would drop it silently."
            )

    @pytest.mark.parametrize("field", LYNX_JOB_FIELDS)
    def test_job_creation_persists_every_lynx_field(self, field):
        tree = ast.parse((ROOT / "app" / "routes" / "jobs.py").read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                    and node.func.id == "Job":
                assert field in {kw.arg for kw in node.keywords if kw.arg}
                return
        pytest.fail("no Job(...) construction found in app/routes/jobs.py")

    @pytest.mark.parametrize("field", LYNX_JOB_FIELDS)
    def test_claim_response_construction_passes_every_lynx_field(self, field):
        tree = ast.parse((ROOT / "app" / "routes" / "segments.py").read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                    and node.func.id == "SegmentClaimResponse":
                assert field in {kw.arg for kw in node.keywords if kw.arg}
                return
        pytest.fail("no SegmentClaimResponse construction found in app/routes/segments.py")


class TestSubjectImageMirroring:
    """The console reuses the starting-image upload slot for the Lynx subject, so
    create_job mirrors the resolved URI across."""

    @staticmethod
    def _mirror_source():
        tree = ast.parse((ROOT / "app" / "routes" / "jobs.py").read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                src = ast.unparse(node)
                if "lynx_subject_image" in src and "generation_engine" in src:
                    return src
        return None

    def test_mirror_exists(self):
        assert self._mirror_source() is not None, (
            "create_job must map starting_image -> lynx_subject_image for Lynx jobs, "
            "or a console-submitted Lynx job reaches the daemon with no subject."
        )

    def test_mirror_only_applies_to_lynx(self):
        assert "'lynx'" in self._mirror_source()

    def test_mirror_does_not_clobber_an_explicit_value(self):
        # guarded by `not job.lynx_subject_image`
        assert "not job.lynx_subject_image" in self._mirror_source()


class TestMigration:
    """Migration 050 must match the model, or a deployed API 500s on a column that
    exists in Python and not in Postgres."""

    @staticmethod
    def _migration_module():
        path = ROOT / "alembic" / "versions" / "050_add_lynx_engine.py"
        namespace: dict = {}
        exec(compile(path.read_text(), str(path), "exec"), namespace)
        return namespace

    def test_revision_chain(self):
        ns = self._migration_module()
        assert ns["revision"] == "050"
        assert ns["down_revision"] == "049"

    def test_job_columns_match_the_model(self):
        declared = [name for name, _ in self._migration_module()["JOB_COLUMNS"]]
        assert declared == LYNX_JOB_FIELDS

    def test_every_migrated_column_exists_on_the_model(self):
        for name, _ in self._migration_module()["JOB_COLUMNS"]:
            assert name in Job.__table__.columns
