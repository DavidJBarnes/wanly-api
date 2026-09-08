"""Character-LoRA training as claimable work (wanly-api#274).

The design decision under test is that training is PULL, like everything else here: the console
creates a row, a trainer claims it. That buys orphan reclaim and the heartbeat sweep for free,
and it only works if the claim gate and the reclaim rules are right — both of which have already
cost this project real incidents in their segment form.
"""
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from app.enums import TRAINING_TERMINAL, TrainingStatus, WorkerKind
from app.models import LtxCharacter, TrainingJob, Worker
from app.routes.training import ORPHANED_TRAINING_MINUTES, _publish_character, _reclaim_orphans
from app.schemas.training import TrainingCreate, TrainingProgress


def _run_sync(coro):
    import asyncio
    return asyncio.run(coro)


def _images(n=13):
    return [f"s3://wanly-images/2026-09-07/img{i}.jpg" for i in range(n)]


def _job(**kw):
    base = dict(id=uuid.uuid4(), character="pay", trigger="p@y", version=1,
                dataset_images=_images(), config={}, status=TrainingStatus.PENDING,
                created_at=datetime.now(timezone.utc))
    base.update(kw)
    return TrainingJob(**base)


def _worker(**kw):
    base = dict(id=uuid.uuid4(), friendly_name="3090/services", hostname="h",
                ip_address="1.2.3.4", kind=WorkerKind.TRAINER, status="online",
                last_heartbeat=datetime.now(timezone.utc))
    base.update(kw)
    return Worker(**base)


class TestTheRequest:
    def test_a_dataset_of_s3_uris_is_required(self):
        """A local path would be meaningless to a trainer on another machine."""
        with pytest.raises(ValueError, match="s3://"):
            TrainingCreate(character="pay", trigger="p@y",
                           dataset_images=["/home/david/img.jpg"] * 13)

    def test_duplicates_are_refused(self):
        """A duplicate trains the same image twice under two sel_NNN names, silently
        reweighting the set toward it."""
        with pytest.raises(ValueError, match="duplicates"):
            TrainingCreate(character="pay", trigger="p@y",
                           dataset_images=["s3://b/a.jpg"] * 13)

    def test_too_few_images_is_refused_by_the_route(self):
        """The floor moved off the schema when dataset_id arrived: a schema minimum on
        dataset_images would reject every dataset_id request, which is the normal path now."""
        import inspect
        from app.routes import training as mod
        assert "at least {MIN_DATASET_IMAGES} are needed" in inspect.getsource(mod.create_training_job)

    def test_a_character_cannot_contain_a_path(self):
        """It becomes a directory name and an output filename on the trainer."""
        for bad in ("../etc", "a/b", ".hidden", "two words"):
            with pytest.raises(ValueError):
                TrainingCreate(character=bad, trigger="t", dataset_images=_images())

    def test_the_trigger_may_differ_from_the_character(self):
        """p@y REQUIRES this: the LoRA is served over HTTP and lands in JSON and URLs, so the
        filename cannot hold an @ — but the captions trained on one."""
        t = TrainingCreate(character="pay", trigger="p@y", dataset_images=_images())
        assert t.character == "pay" and t.trigger == "p@y"


class TestTheClaimGate:
    """Only a trainer may claim. A render worker or a captioner picking up a 50-minute training
    run would be worse than it going unclaimed."""

    @pytest.mark.parametrize("kind", [WorkerKind.RENDER, WorkerKind.SERVICE])
    def test_only_a_trainer_kind_passes(self, kind):
        import inspect
        from app.routes import training as mod
        src = inspect.getsource(mod.claim_next_training_job)
        assert "worker.kind != WorkerKind.TRAINER" in src
        assert kind != WorkerKind.TRAINER

    def test_the_gate_runs_before_the_reclaim_and_the_select(self):
        """Order matters: an ineligible worker must not even trigger a reclaim pass."""
        import inspect
        from app.routes import training as mod
        src = inspect.getsource(mod.claim_next_training_job)
        assert src.index("WorkerKind.TRAINER") < src.index("_reclaim_orphans")

    def test_it_locks_with_skip_locked(self):
        """Two trainers polling at once must not be handed the same row."""
        import inspect
        from app.routes import training as mod
        assert "with_for_update(skip_locked=True)" in inspect.getsource(mod.claim_next_training_job)

    def test_next_is_routed_before_the_id_wildcard(self):
        """/training/{job_id} declared first would swallow /training/next and every poll would
        404 looking for a job called "next"."""
        from app.main import app
        paths = [r.path for r in app.routes if r.path.startswith("/training")]
        assert paths.index("/training/next") < paths.index("/training/{job_id}")


@pytest.mark.asyncio
class TestReclaim:
    async def test_a_claim_from_a_dead_worker_comes_back(self, db):
        w = _worker(last_heartbeat=datetime.now(timezone.utc) - timedelta(minutes=30))
        j = _job(status=TrainingStatus.RUNNING, worker_id=w.id, worker_name=w.friendly_name,
                 claimed_at=datetime.now(timezone.utc) - timedelta(minutes=40),
                 progress_log="[3/4] training")
        db.add_all([w, j])
        await db.flush()
        await _reclaim_orphans(db)
        assert j.status == TrainingStatus.PENDING
        assert j.worker_id is None and j.progress_log is None

    async def test_a_live_worker_making_progress_keeps_its_claim(self, db):
        """The rule that matters. A training run is ~50 minutes; stealing one from a worker that
        is actually training is how two GPUs once converged on the same segment."""
        w = _worker(status="online")
        j = _job(status=TrainingStatus.RUNNING, worker_id=w.id,
                 claimed_at=datetime.now(timezone.utc) - timedelta(hours=2),
                 progress_log="[3/4] step 800/1200")
        db.add_all([w, j])
        await db.flush()
        await _reclaim_orphans(db)
        assert j.status == TrainingStatus.RUNNING

    async def test_a_live_idle_worker_with_no_progress_loses_it(self, db):
        """The lost-claim-response case: the row was assigned before the answer was sent, and
        the answer never arrived. An empty progress log is the only trustworthy evidence."""
        w = _worker(status="online")
        j = _job(status=TrainingStatus.CLAIMED, worker_id=w.id, progress_log=None,
                 claimed_at=datetime.now(timezone.utc)
                 - timedelta(minutes=ORPHANED_TRAINING_MINUTES + 5))
        db.add_all([w, j])
        await db.flush()
        await _reclaim_orphans(db)
        assert j.status == TrainingStatus.PENDING

    async def test_a_recent_claim_is_left_alone(self, db):
        """Staging a dataset and caching latents is legitimately quiet for minutes."""
        w = _worker(status="online")
        j = _job(status=TrainingStatus.CLAIMED, worker_id=w.id, progress_log=None,
                 claimed_at=datetime.now(timezone.utc) - timedelta(minutes=2))
        db.add_all([w, j])
        await db.flush()
        await _reclaim_orphans(db)
        assert j.status == TrainingStatus.CLAIMED


@pytest.mark.asyncio
class TestPublishing:
    async def test_completing_creates_the_character(self, db):
        """GET /loras makes the FILE discoverable, but a pose fills <TRIGGER> from an
        LtxCharacter row — without one the LoRA exists and no recipe can reach it."""
        j = _job(output_lora_path="s3://ltx-loras/character/pay_v1_e05.safetensors")
        db.add(j)
        await db.flush()
        await _publish_character(db, j)
        await db.flush()
        from sqlalchemy import select
        row = (await db.execute(select(LtxCharacter).where(
            LtxCharacter.name == "pay"))).scalar_one()
        assert row.char_lora == "pay_v1_e05"
        assert row.trigger == "p@y"

    async def test_retraining_repoints_the_existing_character(self, db):
        """The whole point of a v2. The strengths are left alone — they may be hand-tuned."""
        db.add(LtxCharacter(name="pay", char_lora="pay_v1_e05.safetensors", trigger="p@y",
                            strength_stage_1=0.9, strength_stage_2=1.4))
        await db.flush()
        j = _job(version=2, output_lora_path="s3://ltx-loras/character/pay_v2_e03.safetensors")
        await _publish_character(db, j)
        await db.flush()
        from sqlalchemy import select
        row = (await db.execute(select(LtxCharacter).where(
            LtxCharacter.name == "pay"))).scalar_one()
        assert row.char_lora == "pay_v2_e03"
        assert row.strength_stage_1 == 0.9, "hand-tuned strengths were overwritten"

    async def test_nothing_is_published_without_a_file(self, db):
        j = _job(output_lora_path=None)
        await _publish_character(db, j)
        from sqlalchemy import select
        assert (await db.execute(select(LtxCharacter))).scalars().first() is None


class TestReporting:
    def test_an_omitted_field_does_not_blank_a_stored_one(self):
        """The mistake the worker heartbeat had to fix twice: a partial report erasing good
        data written by a fuller one."""
        import inspect
        from app.routes import training as mod
        src = inspect.getsource(mod.update_training_job)
        assert "if value is not None:" in src

    def test_a_progress_report_need_not_carry_a_status(self):
        p = TrainingProgress(step=400)
        assert p.status is None and p.progress_log is None

    def test_terminal_states_are_the_three_that_stop_work(self):
        assert TRAINING_TERMINAL == {TrainingStatus.COMPLETED, TrainingStatus.FAILED,
                                     TrainingStatus.CANCELLED}


class TestTheClaimIsAtomicEnough:
    """A claim that mutates and then fails leaves a job owned by a worker that never heard
    about it — wanly-api#242 in a new place. It recovers, but only after the grace period,
    with the queue stopped meanwhile. Cheaper to not create it."""

    def test_presigning_happens_before_the_row_is_marked(self):
        import inspect
        from app.routes import training as mod
        src = inspect.getsource(mod.claim_next_training_job)
        assert src.index("generate_presigned_url") < src.index("job.status = TrainingStatus.CLAIMED")

    def test_a_presign_failure_says_the_job_is_still_queued(self):
        import inspect
        from app.routes import training as mod
        src = inspect.getsource(mod.claim_next_training_job)
        assert "still queued" in src and "503" in src


class TestCharacterNaming:
    """`character` is the LtxCharacter name, `@` and all — not a filesystem-safe version.

    Verified against a real API: training `p@y` v3 repointed the existing `p@y` row from
    `pay_v2_e05` to `pay_v3_e04.safetensors`, one row. Passing `pay` instead created a second
    character row pointing at the same LoRA while every recipe kept using the old one.
    """

    def test_an_at_sign_is_allowed_in_the_character(self):
        t = TrainingCreate(character="p@y", trigger="p@y", dataset_images=_images())
        assert t.character == "p@y"

    def test_the_filename_stem_is_decided_at_creation_not_at_upload(self):
        """It is a field on the request, defaulted and correctable, rather than a guess made
        silently when the file lands. See TestTheLoraFilename for why."""
        import inspect
        from app.routes import training as mod
        assert "lora_name" in inspect.getsource(mod.create_training_job)
        assert 'get("lora_name")' in inspect.getsource(mod._artifact_key)

    def test_the_trigger_never_feeds_the_filename(self):
        """They are different things. The trigger keeps whatever trained."""
        import inspect
        from app.routes import training as mod
        src = inspect.getsource(mod.upload_training_artifact)
        assert "job.trigger" not in src

    @pytest.mark.asyncio
    async def test_retraining_a_character_with_an_at_sign_repoints_one_row(self, db):
        from sqlalchemy import select
        db.add(LtxCharacter(name="p@y", char_lora="pay_v2_e05.safetensors", trigger="p@y"))
        await db.flush()
        j = _job(character="p@y", version=3,
                 output_lora_path="s3://ltx-loras/character/pay_v3_e04.safetensors")
        await _publish_character(db, j)
        await db.flush()
        rows = (await db.execute(select(LtxCharacter).where(
            LtxCharacter.name == "p@y"))).scalars().all()
        assert len(rows) == 1
        assert rows[0].char_lora == "pay_v3_e04"


class TestTheLoraFilename:
    """The stem is ASKED, not derived.

    Stripping `p@y` gives `py`. The file this project has actually been rendering with is
    `pay_v2_e05.safetensors` — a human read `@` as `a`, and no rule produces that. `@`->`a` is a
    transliteration, and a table for it generalises badly: k3lly2026 keeps its digits, so
    `3`->`e` would be wrong.
    """

    def test_the_default_strips_and_never_invents_a_letter(self):
        from app.routes.training import _default_lora_name
        assert _default_lora_name("p@y") == "py"
        assert _default_lora_name("k3lly2026") == "k3lly2026"

    def test_a_name_of_only_unsafe_characters_still_yields_something(self):
        from app.routes.training import _default_lora_name
        assert _default_lora_name("@@@") == "lora"

    def test_an_explicit_name_is_accepted(self):
        t = TrainingCreate(character="p@y", trigger="p@y", lora_name="pay",
                           dataset_images=_images())
        assert t.lora_name == "pay"

    def test_an_unsafe_explicit_name_is_refused(self):
        """Otherwise the field just moves the problem."""
        for bad in ("p@y", "a/b", "with space"):
            with pytest.raises(ValueError):
                TrainingCreate(character="x", trigger="x", lora_name=bad,
                               dataset_images=_images())

    def test_the_upload_uses_the_stored_stem_not_a_fresh_guess(self):
        from app.routes.training import _artifact_key
        job = _job(character="p@y", version=3, config={"lora_name": "pay"})
        assert _artifact_key(job, 4, False) == "character/pay_v3_e04.safetensors"

    def test_both_upload_paths_name_the_file_the_same_way(self):
        """The CLI backfill and the trainer's direct PUT must land on identical keys, or a
        backfilled epoch sits beside the live one under a second name."""
        import inspect
        from app.routes import training as mod
        assert "_artifact_key(job, epoch, final)" in inspect.getsource(mod.upload_training_artifact)
        assert "_artifact_key(job, epoch, final)" in inspect.getsource(mod.presign_training_artifact)


class TestEveryCheckpointIsOffered:
    """Choosing between epochs is a judgement made by eye at a fixed seed. Loss does not rank
    them — a confident "later epochs overfit" call read off a loss curve was refuted outright on
    d0ggyff — so a console that only offers the final epoch has thrown the decision away."""

    def test_uploading_appends_rather_than_replaces(self):
        from app.routes.training import _record_checkpoint
        job = _job(checkpoints=["s3://ltx-loras/character/pay_v1_e01.safetensors"])
        _record_checkpoint(job, "s3://ltx-loras/character/pay_v1_e02.safetensors")
        assert job.checkpoints == ["s3://ltx-loras/character/pay_v1_e01.safetensors",
                                   "s3://ltx-loras/character/pay_v1_e02.safetensors"]

    def test_the_same_uri_is_not_recorded_twice(self):
        from app.routes.training import _record_checkpoint
        job = _job(checkpoints=["s3://ltx-loras/character/pay_v1_e01.safetensors"])
        _record_checkpoint(job, "s3://ltx-loras/character/pay_v1_e01.safetensors")
        assert len(job.checkpoints) == 1

    def test_output_lora_path_tracks_the_most_recent(self):
        """It is what the character row points at until someone picks differently."""
        from app.routes.training import _record_checkpoint
        job = _job()
        _record_checkpoint(job, "s3://ltx-loras/character/pay_v1_e01.safetensors")
        _record_checkpoint(job, "s3://ltx-loras/character/pay_v1_final.safetensors")
        assert job.output_lora_path == "s3://ltx-loras/character/pay_v1_final.safetensors"

    def test_both_upload_paths_record_through_one_function(self):
        import inspect
        from app.routes import training as mod
        assert "_record_checkpoint(job, uri)" in inspect.getsource(mod.upload_training_artifact)
        assert "_record_checkpoint(job, uri)" in inspect.getsource(mod.commit_training_artifact)


class TestTheFinalCheckpointIsNamed:
    """The trainer writes the checkpoint at the end of the step count WITHOUT an epoch number.
    It is usually a partial epoch and it is the one with the most training in it; published
    as a bare `pay_v2.safetensors` it read as "the v2" and hid that four other candidates
    existed. The console labelled it with the whole filename."""

    def test_final_gets_its_own_tag(self):
        from app.routes.training import _artifact_key
        job = _job(character="p@y", version=2, config={"lora_name": "pay"})
        assert _artifact_key(job, None, True) == "character/pay_v2_final.safetensors"

    def test_an_epoch_is_zero_padded(self):
        from app.routes.training import _artifact_key
        assert _artifact_key(_job(config={"lora_name": "pay"}), 5, False).endswith("_e05.safetensors")

    def test_the_url_endpoint_insists_on_one_or_the_other(self):
        """A checkpoint with neither an epoch nor `final` would land on the bare name."""
        from fastapi import HTTPException
        from app.routes.training import presign_training_artifact

        class _DB:
            async def get(self, *_a):
                return _job()

        with pytest.raises(HTTPException) as e:
            _run_sync(presign_training_artifact(uuid.uuid4(), epoch=None, final=False, db=_DB()))
        assert e.value.status_code == 422


class TestACommitIsBelievedOnlyAfterLooking:
    """The trainer PUTs straight to S3 and then says so. Saying so must not be enough: a PUT
    that failed, was truncated, or went to somebody else's key would otherwise put a dead
    download button on the page -- exactly what this replaces."""

    def _job(self):
        return _job(character="p@y", version=2, config={"lora_name": "pay"})

    def test_a_uri_of_another_job_is_refused(self):
        from app.routes.training import _belongs_to
        job = self._job()
        assert not _belongs_to(job, "s3://ltx-loras/character/laura_v2_e01.safetensors")
        assert not _belongs_to(job, "s3://ltx-loras/character/pay_v20_e01.safetensors")
        assert not _belongs_to(job, "s3://ltx-loras/character/pay_v3_e01.safetensors")
        assert not _belongs_to(job, "s3://wanly-images/character/pay_v2_e01.safetensors")

    def test_its_own_names_are_accepted(self):
        from app.routes.training import _belongs_to
        job = self._job()
        assert _belongs_to(job, "s3://ltx-loras/character/pay_v2_e01.safetensors")
        assert _belongs_to(job, "s3://ltx-loras/character/pay_v2_final.safetensors")

    def test_a_put_that_did_not_land_records_nothing(self, monkeypatch):
        from fastapi import HTTPException
        from app.routes import training as mod
        job = self._job()
        monkeypatch.setattr(mod.s3, "head_object", lambda uri: None)

        class _DB:
            async def get(self, *_a):
                return job

        with pytest.raises(HTTPException) as e:
            _run_sync(mod.commit_training_artifact(
                job.id, uri="s3://ltx-loras/character/pay_v2_e01.safetensors", db=_DB()))
        assert e.value.status_code == 409
        assert not job.checkpoints

    def test_a_truncated_object_records_nothing(self, monkeypatch):
        from fastapi import HTTPException
        from app.routes import training as mod
        job = self._job()
        monkeypatch.setattr(mod.s3, "head_object", lambda uri: {"Key": "k", "Size": 1234})

        class _DB:
            async def get(self, *_a):
                return job

        with pytest.raises(HTTPException) as e:
            _run_sync(mod.commit_training_artifact(
                job.id, uri="s3://ltx-loras/character/pay_v2_e01.safetensors", db=_DB()))
        assert e.value.status_code == 409
        assert not job.checkpoints

    async def test_a_real_object_is_recorded(self, db, monkeypatch):
        from app.routes import training as mod
        job = self._job()
        db.add(job)
        await db.commit()
        monkeypatch.setattr(mod.s3, "head_object",
                            lambda uri: {"Key": "k", "Size": 650 * 1024 * 1024})

        out = await mod.commit_training_artifact(
            job.id, uri="s3://ltx-loras/character/pay_v2_final.safetensors", db=db)

        assert out.checkpoints == ["s3://ltx-loras/character/pay_v2_final.safetensors"]
        assert out.output_lora_path == "s3://ltx-loras/character/pay_v2_final.safetensors"


class TestAFinishedRunCanBeDeleted:
    """Four failed and cancelled p@y rows sat above the one that worked, forever."""

    def test_the_files_go_with_it_by_default(self):
        """A deleted run whose checkpoints linger in the library is what someone deleting a
        run does not expect (console#464)."""
        import inspect
        from app.routes.training import delete_training_job
        sig = inspect.signature(delete_training_job)
        assert sig.parameters["purge"].default is True
        assert "s3.delete_object" in inspect.getsource(delete_training_job)

    async def test_a_character_rendering_with_one_stops_the_purge(self, db, monkeypatch):
        """Deleting the file under a character breaks every recipe that names it."""
        from fastapi import HTTPException
        from app.routes import training as mod
        job = _job(status=TrainingStatus.COMPLETED,
                   checkpoints=["s3://ltx-loras/character/pay_v2_final.safetensors"])
        db.add(job)
        db.add(LtxCharacter(name="p@y", char_lora="pay_v2_final", trigger="p@y"))
        await db.commit()
        deleted = []
        monkeypatch.setattr(mod.s3, "delete_object", lambda uri: deleted.append(uri))

        with pytest.raises(HTTPException) as e:
            await mod.delete_training_job(job.id, purge=True, _user=None, db=db)
        assert e.value.status_code == 409
        assert "p@y" in e.value.detail
        assert deleted == []
        assert await db.get(TrainingJob, job.id) is not None

    async def test_a_file_that_cannot_be_deleted_keeps_the_row_and_says_why(self, db, monkeypatch):
        """It 500'd in production: the role had PutObject on character/* and not DeleteObject."""
        from fastapi import HTTPException
        from app.routes import training as mod
        job = _job(status=TrainingStatus.COMPLETED,
                   checkpoints=["s3://ltx-loras/character/pay_v2_final.safetensors"])
        db.add(job)
        await db.commit()

        def denied(uri):
            raise PermissionError("AccessDenied")
        monkeypatch.setattr(mod.s3, "delete_object", denied)
        with pytest.raises(HTTPException) as e:
            await mod.delete_training_job(job.id, purge=True, _user=None, db=db)
        assert e.value.status_code == 503 and "DeleteObject" in e.value.detail
        assert await db.get(TrainingJob, job.id) is not None

    async def test_otherwise_the_files_are_deleted(self, db, monkeypatch):
        from app.routes import training as mod
        uris = ["s3://ltx-loras/character/pay_v2_e01.safetensors",
                "s3://ltx-loras/character/pay_v2_final.safetensors"]
        job = _job(status=TrainingStatus.COMPLETED, checkpoints=uris)
        db.add(job)
        await db.commit()
        deleted = []
        monkeypatch.setattr(mod.s3, "delete_object", lambda uri: deleted.append(uri))

        await mod.delete_training_job(job.id, purge=True, _user=None, db=db)
        assert sorted(deleted) == sorted(uris)
        assert await db.get(TrainingJob, job.id) is None

    def test_a_live_job_is_refused(self):
        from fastapi import HTTPException
        from app.routes.training import delete_training_job
        job = _job(status=TrainingStatus.RUNNING)

        class _DB:
            async def get(self, *_a):
                return job

        with pytest.raises(HTTPException) as e:
            _run_sync(delete_training_job(job.id, purge=True, _user=None, db=_DB()))
        assert e.value.status_code == 409

    async def test_a_terminal_job_goes(self, db):
        from app.routes.training import delete_training_job
        job = _job(status=TrainingStatus.FAILED)
        db.add(job)
        await db.commit()
        await delete_training_job(job.id, purge=False, _user=None, db=db)
        assert await db.get(TrainingJob, job.id) is None


@pytest.mark.asyncio
class TestOnlyDownloadableCheckpointsSurvive:
    """The console turns every `checkpoints` entry into a download button pointed at
    GET /files, which serves an S3 URI and nothing else. Early trainer builds recorded the
    container-local output path instead of uploading, so a completed job carried five entries
    that all 404. Re-uploading has to REPLACE that list, not extend it."""

    async def _upload(self, db, job, monkeypatch, epoch=None):
        import io
        from fastapi import UploadFile
        from app.routes import training as mod

        uploaded = {}

        def fake_upload(data, key, bucket):
            uploaded["key"] = key
            return f"s3://{bucket}/{key}"

        monkeypatch.setattr(mod.s3, "upload_bytes", fake_upload)
        # Over the 10 MB floor the route enforces against truncated uploads.
        payload = b"\0" * (11 * 1024 * 1024)
        f = UploadFile(filename="lora.safetensors", file=io.BytesIO(payload))
        return await mod.upload_training_artifact(job.id, lora=f, epoch=epoch, db=db)

    async def test_a_container_local_path_is_dropped(self, db, monkeypatch):
        job = _job(status=TrainingStatus.COMPLETED, config={"lora_name": "pay"},
                   checkpoints=["/loras/p@y/ltx23b-v2/output/p@y_v2-000003.comfy.safetensors"])
        db.add(job)
        await db.commit()

        out = await self._upload(db, job, monkeypatch, epoch=3)

        assert all(c.startswith("s3://") for c in out.checkpoints), out.checkpoints
        assert len(out.checkpoints) == 1

    async def test_earlier_s3_epochs_are_still_kept(self, db, monkeypatch):
        """Dropping the dead entries must not also drop the good ones — every epoch is
        offered because loss does not rank them."""
        prior = "s3://ltx-loras/character/pay_v2_e01.safetensors"
        job = _job(status=TrainingStatus.COMPLETED, config={"lora_name": "pay"},
                   checkpoints=[prior, "/loras/output/p@y_v2-000002.comfy.safetensors"])
        db.add(job)
        await db.commit()

        out = await self._upload(db, job, monkeypatch, epoch=2)

        assert prior in out.checkpoints
        assert len(out.checkpoints) == 2


@pytest.mark.asyncio
class TestCancellingActuallyCancels:
    """Nothing in the API reaches into the GPU box, so a cancelled job keeps reporting
    `running` until the trainer notices. Writing that report unconditionally undid the cancel
    within seconds: the button appeared to work and the run went to completion."""

    async def _patch(self, db, job, **fields):
        from app.routes.training import update_training_job
        return await update_training_job(job.id, TrainingProgress(**fields), db=db)

    async def test_a_progress_report_does_not_revive_a_cancelled_job(self, db):
        job = _job(status=TrainingStatus.CANCELLED,
                   completed_at=datetime.now(timezone.utc))
        db.add(job)
        await db.commit()

        out = await self._patch(db, job, status=TrainingStatus.RUNNING, step=412)

        assert out.status == TrainingStatus.CANCELLED
        # The progress itself is still recorded — it is what the run was doing when it stopped.
        assert out.step == 412

    async def test_a_cancelled_job_is_not_completed_by_a_late_finish(self, db):
        """A trainer that finishes before it notices has not made the run wanted again."""
        job = _job(status=TrainingStatus.CANCELLED,
                   completed_at=datetime.now(timezone.utc))
        db.add(job)
        await db.commit()

        out = await self._patch(db, job, status=TrainingStatus.COMPLETED)

        assert out.status == TrainingStatus.CANCELLED

    async def test_the_reply_tells_the_trainer_it_was_cancelled(self, db):
        """This is how the trainer finds out — there is no second call."""
        job = _job(status=TrainingStatus.CANCELLED,
                   completed_at=datetime.now(timezone.utc))
        db.add(job)
        await db.commit()

        out = await self._patch(db, job, status=TrainingStatus.RUNNING)

        assert out.status == TrainingStatus.CANCELLED

    async def test_an_ordinary_running_job_still_advances(self, db):
        job = _job(status=TrainingStatus.CLAIMED)
        db.add(job)
        await db.commit()

        out = await self._patch(db, job, status=TrainingStatus.RUNNING, step=7)

        assert out.status == TrainingStatus.RUNNING
        assert out.step == 7


class TestOnlyTheFinalGoesUpByDefault:
    """A 650 MB checkpoint takes ~18 minutes to leave the 3090 and "I often only want 1 or 2
    epochs". Every epoch stays on the trainer; the rest are asked for."""

    def test_the_default_policy_is_final(self):
        req = TrainingCreate(character="p@y", trigger="p@y", dataset_images=_images())
        assert req.publish == "final"

    def test_all_is_the_other_choice_and_nothing_else_is(self):
        assert TrainingCreate(character="p@y", trigger="p@y", dataset_images=_images(),
                              publish="all").publish == "all"
        with pytest.raises(ValueError):
            TrainingCreate(character="p@y", trigger="p@y", dataset_images=_images(),
                           publish="some")

    def test_the_policy_reaches_the_job_config(self):
        import inspect
        from app.routes import training as mod
        assert '"publish": body.publish' in inspect.getsource(mod.create_training_job)

    async def test_a_publish_request_is_recorded_once(self, db):
        from app.routes.training import request_publish
        job = _job(status=TrainingStatus.COMPLETED,
                   epochs=[{"label": "e01", "step": 80, "loss": 0.7},
                           {"label": "final", "step": 100, "loss": 0.68}],
                   checkpoints=["s3://ltx-loras/character/pay_v1_final.safetensors"])
        db.add(job)
        await db.commit()
        out = await request_publish(job.id, label="e01", _user=None, db=db)
        out = await request_publish(job.id, label="e01", _user=None, db=db)
        assert out.publish_requests == ["e01"]

    async def test_a_checkpoint_the_run_never_wrote_is_refused(self, db):
        from fastapi import HTTPException
        from app.routes.training import request_publish
        job = _job(status=TrainingStatus.COMPLETED, epochs=[{"label": "final", "step": 100}])
        db.add(job)
        await db.commit()
        with pytest.raises(HTTPException) as e:
            await request_publish(job.id, label="e07", _user=None, db=db)
        assert e.value.status_code == 404

    async def test_one_already_in_the_bucket_is_refused(self, db):
        from fastapi import HTTPException
        from app.routes.training import request_publish
        job = _job(status=TrainingStatus.COMPLETED, epochs=[{"label": "final", "step": 100}],
                   checkpoints=["s3://ltx-loras/character/pay_v1_final.safetensors"])
        db.add(job)
        await db.commit()
        with pytest.raises(HTTPException) as e:
            await request_publish(job.id, label="final", _user=None, db=db)
        assert e.value.status_code == 409


class TestThePublishedCharacterStoresTheStem:
    async def test_no_extension_on_the_row(self, db):
        """Every other row stores `pay_v2_e05`; the console compares stems."""
        job = _job(character="Me", output_lora_path="s3://ltx-loras/character/david_v1_final.safetensors")
        await _publish_character(db, job)
        await db.commit()
        from sqlalchemy import select
        c = (await db.execute(select(LtxCharacter).where(LtxCharacter.name == "Me"))).scalar_one()
        assert c.char_lora == "david_v1_final"


class TestTheLoraHasAFace:
    async def test_the_dataset_anchor_is_snapshotted_at_creation(self, db):
        from app.models import Dataset
        from app.routes.training import create_training_job
        ds = Dataset(name="faces", images=_images(), anchor_uri=_images()[3], prefix="x")
        db.add(ds)
        await db.commit()

        class _U:
            id = None
            username = "t"
        job = await create_training_job(
            TrainingCreate(character="p@y", trigger="p@y", dataset_id=ds.id), user=_U(), db=db)
        assert job.thumbnail_uri == _images()[3]

    async def test_publishing_puts_the_face_on_the_character(self, db):
        job = _job(character="p@y", output_lora_path="s3://ltx-loras/character/pay_v3_final.safetensors",
                   thumbnail_uri="s3://wanly-images/datasets/x/anchor.jpg")
        await _publish_character(db, job)
        await db.commit()
        from sqlalchemy import select
        c = (await db.execute(select(LtxCharacter).where(LtxCharacter.name == "p@y"))).scalar_one()
        assert c.image_uri == "s3://wanly-images/datasets/x/anchor.jpg"

    def test_a_progress_report_can_carry_the_curve_and_the_epochs(self):
        p = TrainingProgress(loss_log=[[10, 0.9], [20, 0.8]],
                             epochs=[{"label": "e01", "step": 80, "loss": 0.7}])
        assert p.loss_log[-1] == [20, 0.8]
        import inspect
        from app.routes import training as mod
        assert '"loss_log", "epochs"' in inspect.getsource(mod.update_training_job)


class TestTheCaptionCarriesTheTrigger:
    """Me v1 trained with the caption "man": the trigger d@vid was never in a caption, so the
    model never learned it and the identity bound to "man" instead."""

    async def _create(self, db, caption):
        from app.routes.training import create_training_job

        class _U:
            id = None
            username = "t"
        return await create_training_job(
            TrainingCreate(character="Me", trigger="d@vid", dataset_images=_images(),
                           caption=caption), user=_U(), db=db)

    async def test_a_caption_without_the_trigger_gets_it_prepended(self, db):
        job = await self._create(db, "man")
        assert job.config["caption"] == "d@vid, man"

    async def test_a_caption_that_names_it_is_left_alone(self, db):
        job = await self._create(db, "portrait of d@vid, man")
        assert job.config["caption"] == "portrait of d@vid, man"

    async def test_no_caption_stays_none_for_the_trainers_default(self, db):
        job = await self._create(db, "   ")
        assert job.config["caption"] is None

    async def test_gender_writes_the_whole_caption(self, db):
        from app.routes.training import create_training_job

        class _U:
            id = None
            username = "t"
        job = await create_training_job(
            TrainingCreate(character="Me", trigger="d@vid", dataset_images=_images(),
                           gender="man"), user=_U(), db=db)
        assert job.config["caption"] == "d@vid, man"
        assert job.config["gender"] == "man"

    def test_gender_is_one_of_three(self):
        with pytest.raises(ValueError):
            TrainingCreate(character="Me", trigger="d@vid", dataset_images=_images(), gender="boy")
