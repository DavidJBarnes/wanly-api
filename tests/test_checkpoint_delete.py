"""Deleting one checkpoint forever (wanly-api#413).

"Forever" cannot be taken back, so most of these are about what must STOP a delete: a live
run, a character rendering with it, a segment still to render with it -- and that a refused
delete changes nothing, in the bucket or on the row.
"""
import uuid
from datetime import datetime, timezone

import pytest

from app.enums import SegmentStatus, TrainingStatus
from app.models import Job, LtxCharacter, Segment, TrainingJob, User

FINAL = "s3://ltx-loras/character/pay_v1_final.safetensors"
E02 = "s3://ltx-loras/character/pay_v1_e02.safetensors"
EPOCHS = [{"label": "e01", "step": 40, "loss": 0.8}, {"label": "e02", "step": 80, "loss": 0.7},
          {"label": "final", "step": 100, "loss": 0.68}]


def _job(**kw):
    base = dict(id=uuid.uuid4(), character="pay", trigger="p@y", version=1,
                dataset_images=["s3://wanly-images/a.jpg"], config={"publish": "none"},
                status=TrainingStatus.COMPLETED, created_at=datetime.now(timezone.utc),
                epochs=list(EPOCHS), checkpoints=[E02, FINAL], output_lora_path=FINAL)
    base.update(kw)
    return TrainingJob(**base)


@pytest.fixture
def bucket(monkeypatch):
    from app.routes import training as mod
    deleted = []
    monkeypatch.setattr(mod.s3, "delete_object", lambda uri: deleted.append(uri))
    return deleted


async def _delete(db, job, label):
    from app.routes.training import delete_checkpoint
    return await delete_checkpoint(job.id, label, _user=None, db=db)


@pytest.mark.asyncio
class TestDelete:
    async def test_an_uploaded_checkpoint_goes_from_the_bucket_and_the_row(self, db, bucket):
        job = _job(publish_requests=["e02"])
        db.add(job)
        await db.commit()
        out = await _delete(db, job, "e02")
        assert bucket == [E02]
        assert out.checkpoints == [FINAL] and out.output_lora_path == FINAL
        assert [e["label"] for e in out.epochs] == ["e01", "final"]
        assert out.publish_requests is None
        assert out.delete_requests == ["e02"]          # for the trainer's disk

    async def test_a_checkpoint_only_on_the_trainer_is_just_queued_for_its_disk(self, db, bucket):
        job = _job()
        db.add(job)
        await db.commit()
        out = await _delete(db, job, "e01")
        assert bucket == []
        assert out.checkpoints == [E02, FINAL] and out.delete_requests == ["e01"]

    async def test_deleting_the_final_repoints_the_output_at_what_is_left(self, db, bucket):
        job = _job()
        db.add(job)
        await db.commit()
        out = await _delete(db, job, "final")
        assert out.output_lora_path == E02 and out.checkpoints == [E02]

    async def test_twice_is_a_404_not_a_second_request(self, db, bucket):
        from fastapi import HTTPException
        job = _job()
        db.add(job)
        await db.commit()
        await _delete(db, job, "e01")
        with pytest.raises(HTTPException) as e:
            await _delete(db, job, "e01")
        assert e.value.status_code == 404

    async def test_a_live_run_is_refused(self, db, bucket):
        from fastapi import HTTPException
        job = _job(status=TrainingStatus.RUNNING)
        db.add(job)
        await db.commit()
        with pytest.raises(HTTPException) as e:
            await _delete(db, job, "e01")
        assert e.value.status_code == 409 and bucket == []

    async def test_a_character_rendering_with_it_stops_it(self, db, bucket):
        from fastapi import HTTPException
        job = _job()
        db.add_all([job, LtxCharacter(name="Pay", trigger="p@y", char_lora="pay_v1_final")])
        await db.commit()
        with pytest.raises(HTTPException) as e:
            await _delete(db, job, "final")
        assert e.value.status_code == 409 and "Pay renders with pay_v1_final" in e.value.detail
        assert bucket == []
        await db.refresh(job)
        assert job.checkpoints == [E02, FINAL] and job.delete_requests is None

    async def test_a_queued_segment_naming_it_stops_it_a_finished_one_does_not(self, db, bucket):
        from fastapi import HTTPException
        job = _job()
        user = User(username=str(uuid.uuid4()), password_hash="x")
        db.add_all([job, user])
        await db.flush()
        render = Job(user_id=user.id, name="j", width=832, height=1216, fps=24, seed=7,
                     starting_image="s3://wanly-images/start.jpg")
        db.add(render)
        await db.flush()
        recipe = {"characters": [{"name": "Pay", "char_lora": "pay_v1_e02"}]}
        done = Segment(job_id=render.id, index=0, prompt="p", status=SegmentStatus.COMPLETED,
                       ltx_recipe=recipe)
        db.add(done)
        await db.commit()
        out = await _delete(db, job, "e02")          # a finished segment is only a record
        assert bucket == [E02]

        job2 = _job(id=uuid.uuid4(), version=2,
                    checkpoints=["s3://ltx-loras/character/pay_v2_e02.safetensors"])
        queued = Segment(job_id=render.id, index=1, prompt="p", status=SegmentStatus.PENDING,
                         ltx_recipe={"characters": [{"char_lora": "pay_v2_e02"}]})
        db.add_all([job2, queued])
        await db.commit()
        with pytest.raises(HTTPException) as e:
            await _delete(db, job2, "e02")
        assert e.value.status_code == 409 and "1 queued or running segment" in e.value.detail

    async def test_a_bucket_failure_changes_nothing(self, db, monkeypatch):
        from fastapi import HTTPException
        from app.routes import training as mod

        def denied(uri):
            raise PermissionError("AccessDenied")
        monkeypatch.setattr(mod.s3, "delete_object", denied)
        job = _job()
        db.add(job)
        await db.commit()
        with pytest.raises(HTTPException) as e:
            await _delete(db, job, "e02")
        assert e.value.status_code == 503
        await db.refresh(job)
        assert job.checkpoints == [E02, FINAL] and job.delete_requests is None

    async def test_a_deleted_checkpoint_cannot_be_published_again(self, db, bucket):
        from fastapi import HTTPException
        from app.routes.training import request_publish
        job = _job()
        db.add(job)
        await db.commit()
        await _delete(db, job, "e01")
        with pytest.raises(HTTPException) as e:
            await request_publish(job.id, label="e01", _user=None, db=db)
        assert e.value.status_code == 404


@pytest.mark.asyncio
class TestTheUploadRace:
    async def test_a_commit_of_a_deleted_label_is_refused_and_its_object_removed(
            self, db, bucket, monkeypatch):
        """Deleted while the trainer's PUT was in flight: recording it would bring it back
        with no row to delete it from."""
        from fastapi import HTTPException
        from app.routes.training import commit_training_artifact
        job = _job(checkpoints=[FINAL], delete_requests=["e01"])
        db.add(job)
        await db.commit()
        uri = "s3://ltx-loras/character/pay_v1_e01.safetensors"
        with pytest.raises(HTTPException) as e:
            await commit_training_artifact(job.id, uri=uri, db=db)
        assert e.value.status_code == 410 and bucket == [uri]


class TestTheFinalKeepsTheOutput:
    def test_an_epoch_committed_after_the_final_does_not_take_it(self):
        """Under "all" the final goes up first and the queued epochs follow it."""
        from app.routes.training import _record_checkpoint
        job = _job(checkpoints=None, output_lora_path=None)
        _record_checkpoint(job, FINAL)
        _record_checkpoint(job, E02)
        assert job.output_lora_path == FINAL

    def test_without_a_final_the_latest_epoch_still_leads(self):
        from app.routes.training import _record_checkpoint
        job = _job(checkpoints=None, output_lora_path=None)
        _record_checkpoint(job, "s3://ltx-loras/character/pay_v1_e01.safetensors")
        _record_checkpoint(job, E02)
        assert job.output_lora_path == E02


@pytest.mark.asyncio
async def test_retry_forgets_the_delete_requests():
    """The retry writes the same version-named files; an old request would delete the new
    attempt's checkpoint."""
    import inspect
    from app.routes import training as mod
    assert "job.delete_requests = None" in inspect.getsource(mod.retry_training_job)
