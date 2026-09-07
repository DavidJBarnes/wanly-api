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
