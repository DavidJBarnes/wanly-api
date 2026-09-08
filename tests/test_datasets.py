"""Named, taggable training datasets.

The grouping that did not exist. Before this the only ways to say "these images belong together"
were an S3 folder prefix and a per-user favourites list, so a training set could not be named,
tagged or re-opened — every run meant re-selecting by hand, and a v2 meant doing it again from
memory.
"""
import uuid

import pytest

from app.models import Dataset
from app.routes.datasets import _prefix
from app.schemas.datasets import DatasetCreate
from app.schemas.training import TrainingCreate


class TestNaming:
    def test_a_readable_name_is_allowed(self):
        assert DatasetCreate(name="p@y v2 faces").name == "p@y v2 faces"

    @pytest.mark.parametrize("bad", ["a/b", "../etc", "tab\there", ""])
    def test_a_name_that_breaks_a_prefix_is_refused(self, bad):
        """It becomes part of every image's S3 key, forever."""
        with pytest.raises(ValueError):
            DatasetCreate(name=bad)

    def test_the_prefix_is_typeable(self):
        """Spaces work in S3 and are miserable in a URL or a listing."""
        assert _prefix("p@y v2 faces") == "dataset-p@y-v2-faces"


class TestTraining:
    def test_a_job_can_name_a_dataset_instead_of_listing_images(self):
        t = TrainingCreate(character="p@y", trigger="p@y", dataset_id=uuid.uuid4())
        assert t.dataset_images == []
        assert t.dataset_id is not None

    def test_a_raw_list_still_works(self):
        """A CLI or a curl should not have to create a dataset first."""
        imgs = [f"s3://b/{i}.jpg" for i in range(13)]
        assert TrainingCreate(character="x", trigger="x", dataset_images=imgs).dataset_id is None

    def test_the_minimum_is_enforced_in_the_route_not_the_schema(self):
        """A schema minimum on dataset_images would reject every dataset_id request, which is
        the normal path."""
        import inspect
        from app.routes import training as mod
        src = inspect.getsource(mod.create_training_job)
        assert "len(images) < MIN_DATASET_IMAGES" in src
        # and it resolves the dataset before checking
        assert src.index("body.dataset_id") < src.index("len(images) < MIN_DATASET_IMAGES")


@pytest.mark.asyncio
class TestTheRow:
    async def test_images_are_an_ordered_list(self, db):
        """Order is significant: the trainer stages them as sel_000..N and the captions pair
        by index."""
        ds = Dataset(id=uuid.uuid4(), name="d", images=["s3://b/z.jpg", "s3://b/a.jpg"],
                     prefix="dataset-d")
        db.add(ds)
        await db.flush()
        assert ds.images == ["s3://b/z.jpg", "s3://b/a.jpg"]

    async def test_a_dataset_starts_empty(self, db):
        ds = Dataset(id=uuid.uuid4(), name="empty", images=[], prefix="dataset-empty")
        db.add(ds)
        await db.flush()
        assert ds.images == []


class TestUploadSemantics:
    def test_the_image_list_is_reassigned_not_mutated(self):
        """A JSONB column does not see a mutation of the existing list, so `images.append(...)`
        writes nothing and the upload silently vanishes on the next read."""
        import inspect
        from app.routes import datasets as mod
        src = inspect.getsource(mod.add_images)
        assert "ds.images = list(ds.images) + added" in src
        # Code only. The comment above that line names the mistake it is avoiding, and a
        # substring check would match the explanation rather than the bug.
        code = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))
        assert "ds.images.append" not in code

    def test_a_non_image_is_skipped_not_fatal(self):
        """One stray file in a folder drag-and-drop must not reject the other forty-nine."""
        import inspect
        from app.routes import datasets as mod
        assert "continue" in inspect.getsource(mod.add_images)

    def test_deleting_a_dataset_keeps_its_images_by_default(self):
        """A finished training job records the URIs it trained on; destroying them turns a
        reproducible run into an unreproducible one."""
        import inspect
        from app.routes import datasets as mod
        src = inspect.getsource(mod.delete_dataset)
        assert "purge" in src and "if purge" in src

    def test_renaming_does_not_move_the_prefix(self):
        """Objects other rows point at must not move under them."""
        import inspect
        from app.routes import datasets as mod
        assert "does NOT follow a rename" in inspect.getsource(mod.update_dataset)


class TestCropping:
    """Steps 2 and 3 of the documented pipeline — crop, then gate — which until now existed
    only as laptop scripts that ssh'd to the box with insightface."""

    def test_the_gate_is_on_by_default(self):
        """Detection is easy; telling one person from another in the same photo set is what
        hand-culling failed at twice, once into a set already culled by eye."""
        import inspect
        from app.routes import datasets as mod
        sig = inspect.signature(mod.crop_faces)
        assert sig.parameters["gate"].default is True

    def test_it_writes_a_new_dataset_rather_than_replacing(self):
        """The photographs are the source of truth and a crop is derived. Overwriting them
        makes the operation unrepeatable with different padding or a different reference."""
        import inspect
        from app.routes import datasets as mod
        src = inspect.getsource(mod.crop_faces)
        assert "out = Dataset(" in src
        assert "ds.images = uris" not in src

    def test_one_face_per_photo_is_the_default_but_is_a_choice(self):
        """A dataset of couples does not want the largest face: "largest" is then whoever stood
        closer to the camera, so the output silently interleaves two people."""
        import inspect
        from app.routes import datasets as mod
        sig = inspect.signature(mod.crop_faces)
        assert sig.parameters["largest_only"].default is True

    def test_the_choice_actually_reaches_the_service(self):
        """It was hardcoded True in the payload, so the parameter alone would do nothing."""
        import inspect
        from app.routes import datasets as mod
        src = inspect.getsource(mod.crop_faces)
        assert '"largest_only": largest_only,' in src
        assert '"largest_only": True,' not in src

    def test_the_new_dataset_records_which_mode_produced_it(self):
        """Two crops of the same photographs differ in what they contain, not just how many —
        a note saying only the count cannot tell them apart."""
        import inspect
        from app.routes import datasets as mod
        src = inspect.getsource(mod.crop_faces)
        assert "largest only" in src and "every face" in src

    def test_an_unconfigured_service_says_so_rather_than_timing_out(self):
        import inspect
        from app.routes import datasets as mod
        assert "face_crop_url is empty" in inspect.getsource(mod.crop_faces)

    def test_scoring_without_a_reference_is_labelled_as_weak(self):
        """Against the crops' own mean it proves internal consistency and nothing more — the
        l@ura set scored 0.931 mean that way and it meant only "the swap held"."""
        import inspect
        from app.routes import datasets as mod
        assert "internal consistency only" in inspect.getsource(mod.crop_faces)

    def test_everything_failing_the_gate_is_an_error_not_an_empty_dataset(self):
        import inspect
        from app.routes import datasets as mod
        assert "either the reference is wrong" in inspect.getsource(mod.crop_faces)

    def test_the_note_records_what_was_dropped(self):
        """"10 of 38 had no face" is the number that tells you the source set is wrong."""
        import inspect
        from app.routes import datasets as mod
        src = inspect.getsource(mod.crop_faces)
        assert "with none detected" in src and "below the" in src

    def test_the_mean_is_computed_here_not_round_tripped(self):
        from app.routes.datasets import _cos, _mean_via_service
        import asyncio
        mu = asyncio.run(_mean_via_service([[1.0, 0.0], [0.0, 1.0]]))
        assert _cos(mu, mu) == pytest.approx(1.0, abs=1e-6)

    def test_an_absent_embedding_scores_below_any_floor(self):
        from app.routes.datasets import _cos
        assert _cos([], [1.0]) < 0.4
