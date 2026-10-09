"""wanly-api#445: which image in a set was made from which."""
from types import SimpleNamespace

import pytest

from app import lineage
from app.backfill_lineage import by_name, plan

P = "s3://wanly-images/datasets/abc"


def _ds(images, derived=None):
    return SimpleNamespace(images=list(images), derived=derived)


class TestRecordAndPrune:
    def test_record_reassigns_and_never_points_at_itself(self):
        ds = _ds([f"{P}/a.png", f"{P}/b.png"])
        lineage.record(ds, f"{P}/b.png", f"{P}/a.png", "crop", at="t")
        lineage.record(ds, f"{P}/a.png", f"{P}/a.png", "crop")
        assert ds.derived == {f"{P}/b.png": {"from": f"{P}/a.png", "how": "crop", "at": "t"}}

    def test_unknown_how_is_refused(self):
        with pytest.raises(ValueError):
            lineage.record(_ds([]), "x", "y", "magic")

    def test_prune_keeps_an_entry_whose_source_left(self):
        ds = _ds([f"{P}/crop.png"], {f"{P}/crop.png": {"from": f"{P}/gone.png", "how": "crop"},
                                     f"{P}/removed.png": {"from": f"{P}/x.png", "how": "edit"}})
        lineage.prune(ds)
        assert list(ds.derived) == [f"{P}/crop.png"]


class TestNames:
    @pytest.mark.parametrize("uri,expect", [
        (f"{P}/faces-1a2b3c/003_sel_012_f0.jpg", ("sel_012", "crop")),
        (f"{P}/portraits-1a2b3c/000_sel_012_f1.png", ("sel_012", "crop")),
        (f"{P}/portraits-fix123/004_sel_012.jpg", ("sel_012", "fix_crop")),
        (f"{P}/pairs-fix123/004_sel_012.jpg", ("sel_012", "fix_crop")),
        (f"{P}/upscaled-fix123/002_sel_012.jpg", ("sel_012", "upscale")),
        (f"{P}/edits/sel_012_edit-smile_1a2b3c.png", ("sel_012", "edit")),
        (f"{P}/sel_012.png", None),
    ])
    def test_by_name(self, uri, expect):
        assert by_name(uri) == expect


class TestPlan:
    def test_names_link_to_the_one_source_with_that_stem(self):
        orig = f"{P}/sel_012.png"
        crop = f"{P}/faces-1a2b3c/003_sel_012_f0.jpg"
        edit = f"{P}/edits/sel_012_edit-smile_1a2b3c.png"
        out = plan([orig, crop, edit], {}, {}, {})
        assert out == {crop: (orig, "crop"), edit: (orig, "edit")}

    def test_an_ambiguous_stem_records_nothing(self):
        a, b = f"{P}/x/sel_1.png", f"{P}/y/sel_1.png"
        crop = f"{P}/faces-1a2b3c/000_sel_1_f0.jpg"
        assert crop not in plan([a, b, crop], {}, {}, {})

    def test_fix_bookkeeping_wins_and_existing_entries_are_kept(self):
        orig, crop, up, small = (f"{P}/a.png", f"{P}/c.jpg", f"{P}/u.jpg", f"{P}/small.png")
        faces = {orig: {"crop_uri": crop}, up: {"upscaled_from": small}}
        out = plan([orig, crop, up], faces, {crop: {"from": "?", "how": "crop"}}, {})
        assert out == {up: (small, "upscale")}

    def test_identical_bytes_are_a_duplicate_of_the_first(self):
        a, b = f"{P}/a.png", f"{P}/b.png"
        assert plan([a, b], {}, {}, {a: "e1", b: "e1"}) == {b: (a, "duplicate")}

    def test_shared_hex_picks_the_one_plain_member_as_original(self):
        h = "0123456789abcdef0123456789abcdef"
        orig, other = f"{P}/{h}.png", f"{P}/{h}_v2.png"
        # Two plain members: no single original, nothing recorded.
        assert plan([orig, other], {}, {}, {}) == {}
        crop = f"{P}/faces-zz/000_{h}_f0.jpg"
        out = plan([orig, crop], {}, {}, {})
        assert out == {crop: (orig, "crop")}

    def test_a_multipart_etag_is_not_a_content_hash(self):
        a, b = f"{P}/a.png", f"{P}/b.png"
        assert plan([a, b], {}, {}, {a: "e-2", b: "e-2"}) == {}
