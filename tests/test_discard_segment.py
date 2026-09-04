"""Discarding keeps the take and removes the segment from the video.

Deleting destroys the row and with it the record that this seed produced this clip. A bad take
is frequently the most informative one, and destroying it in order to get it out of the cut is
exactly backwards -- so the row survives, keeping its index and its output.
"""

import os

os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://x/y")
os.environ.setdefault("JWT_SECRET", "x" * 32)

from app.models import Segment


class TestSchema:
    def test_segments_carry_a_discarded_flag(self):
        assert "discarded" in Segment.__table__.c

    def test_it_defaults_to_live(self):
        # A column defaulting the other way would hide every existing segment on deploy.
        assert Segment.__table__.c.discarded.server_default.arg.lower() == "false"

    def test_the_unique_index_is_partial(self):
        """This is what lets a discarded segment 2 and its replacement both be 'segment 2'.

        The discarded row keeps its index so the record reads correctly; the regenerated one
        takes the same position in the video. Under a full constraint the replacement would have
        to be appended at the end and would play out of order.
        """
        idx = next(i for i in Segment.__table__.indexes
                   if i.name == "uq_segments_job_index_live")
        assert idx.unique
        assert [c.name for c in idx.columns] == ["job_id", "index"]
        where = idx.dialect_options["postgresql"].get("where")
        assert where is not None and "discarded" in str(where)

    def test_the_old_full_constraint_is_gone(self):
        # Leaving it would defeat the partial index entirely.
        names = {c.name for c in Segment.__table__.constraints if c.name}
        assert "uq_segments_job_index" not in names


class TestVideoExcludesDiscarded:
    def test_stitch_filters_on_discarded(self):
        # The filter has to be explicit: a discarded segment keeps status COMPLETED and its
        # index, so selecting on status alone would still put it in the cut -- the exact thing
        # the user asked to remove.
        import inspect

        from app import stitch

        src = inspect.getsource(stitch)
        assert "Segment.discarded.is_(False)" in src
