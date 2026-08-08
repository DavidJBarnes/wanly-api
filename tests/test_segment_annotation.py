"""Tests for segment annotation (human observations).

The point of these fields is that the automated metrics cannot rank quality -- on the one
controlled pair where both exist, the expression score ranked them backwards, because a gaping
mouth is the largest landmark excursion a face can make and the metric rewards it (daemon#123).

So the valuable property is that tags GROUP. A vocabulary that admits near-misses produces
"mouth-void" and "mouth void" as separate labels and destroys the only thing they are for.
"""

import pytest

from app.schemas.segments import EXCLUSIVE_TAG_GROUPS, OBSERVATION_TAGS, SegmentAnnotation


class TestVocabulary:
    def test_covers_the_artifacts_actually_observed(self):
        # Named from real reports, not invented: the open-mouth failure was described as
        # "blank/black space mouth" with "messy teeth and lips".
        for tag in ("mouth-void", "teeth-mush", "face-frozen"):
            assert tag in OBSERVATION_TAGS

    def test_carries_positive_labels_too(self):
        # Rating is one all-in number, so per-axis signal has to come from somewhere. Positive
        # tags are what let "the motion was good but the mouth was a void" survive as data.
        for tag in ("her-strong", "face-expressive"):
            assert tag in OBSERVATION_TAGS

    def test_motion_is_labelled_per_person(self):
        # motion_magnitude is whole-frame optical flow -- it sums every moving pixel, so a lively
        # woman with a static man scores the same as the reverse. The metric cannot tell them
        # apart even in principle, so these tags are the only per-person motion data obtainable.
        for tag in ("him-strong", "her-strong", "him-static", "her-static"):
            assert tag in OBSERVATION_TAGS

    def test_no_duplicates(self):
        assert len(OBSERVATION_TAGS) == len(set(OBSERVATION_TAGS))

    def test_every_tag_follows_location_condition(self):
        # The scheme is what keeps the set extensible without renaming debates: a new observation
        # about the eyes is eyes-<something>. A tag that does not split cannot be grouped by
        # region, which is how the chip row is laid out and how analysis will slice it.
        for tag in OBSERVATION_TAGS:
            location, _, condition = tag.partition("-")
            assert location and condition, f"{tag} is not <location>-<condition>"
            assert tag.islower() and " " not in tag


class TestAnnotationSchema:
    def test_rating_is_bounded(self):
        with pytest.raises(Exception):
            SegmentAnnotation(rating=0)
        with pytest.raises(Exception):
            SegmentAnnotation(rating=6)
        assert SegmentAnnotation(rating=3).rating == 3

    def test_everything_is_optional_so_a_partial_save_is_possible(self):
        # Saving a rating must not blank out notes typed earlier.
        a = SegmentAnnotation(rating=4)
        assert a.model_dump(exclude_unset=True) == {"rating": 4}

    def test_an_explicit_null_is_distinguishable_from_an_absent_field(self):
        # This is the difference between "leave notes alone" and "clear my notes".
        assert "notes" in SegmentAnnotation(notes=None).model_dump(exclude_unset=True)
        assert "notes" not in SegmentAnnotation(rating=2).model_dump(exclude_unset=True)


class TestTagOrdering:
    def test_storage_order_is_vocabulary_order_not_input_order(self):
        # Two segments tagged the same way must produce identical strings, or grouping needs
        # normalising at read time and every later analysis has to remember to do it.
        chosen = {"her-strong", "mouth-void"}
        ordered = [t for t in OBSERVATION_TAGS if t in chosen]
        reversed_input = {"mouth-void", "her-strong"}
        assert ordered == [t for t in OBSERVATION_TAGS if t in reversed_input]


class TestPace:
    def test_pace_has_an_explicit_middle(self):
        # Every other axis can leave "fine" implicit by being untagged. Pace cannot: with only
        # fast and slow, an untagged segment is ambiguous between "the pace was right" and "I
        # did not judge the pace", and those are different data.
        for tag in ("pace-slow", "pace-right", "pace-fast"):
            assert tag in OBSERVATION_TAGS


class TestContradictoryTags:
    """Tags exist to be ground truth, so a contradictory label is worse than a missing one --
    it quietly poisons whatever it is later used to validate."""

    def test_pace_values_are_mutually_exclusive(self):
        group = next(g for g in EXCLUSIVE_TAG_GROUPS if "pace-fast" in g)
        assert {"pace-slow", "pace-right", "pace-fast"} == group

    def test_a_person_cannot_be_static_and_strong(self):
        assert {"him-static", "him-strong"} in EXCLUSIVE_TAG_GROUPS
        assert {"her-static", "her-strong"} in EXCLUSIVE_TAG_GROUPS

    def test_compatible_face_tags_are_not_restricted(self):
        # A face can be blurry AND frozen. Over-constraining would lose real observations.
        for group in EXCLUSIVE_TAG_GROUPS:
            assert not {"face-blurry", "face-frozen"} <= group

    def test_every_exclusive_member_is_in_the_vocabulary(self):
        for group in EXCLUSIVE_TAG_GROUPS:
            for tag in group:
                assert tag in OBSERVATION_TAGS


class TestBodyMechanics:
    def test_the_condition_carries_the_judgement(self):
        # bodies-unison described the behaviour without saying it was wrong, unlike every other
        # tag. "rocking" is the reviewer's own word for the failure and reads as a defect.
        assert "bodies-rocking" in OBSERVATION_TAGS
        assert "bodies-unison" not in OBSERVATION_TAGS

    def test_impact_is_labelled_too(self):
        # Same reason pace-right exists: without it, untagged is ambiguous between "the bodies
        # impacted properly" and "I did not judge the mechanics".
        assert "bodies-impact" in OBSERVATION_TAGS

    def test_rocking_and_impact_are_mutually_exclusive(self):
        assert {"bodies-rocking", "bodies-impact"} in EXCLUSIVE_TAG_GROUPS

    def test_rocking_is_not_exclusive_with_strong_motion(self):
        # Both people CAN be moving strongly and still be moving wrongly -- that is exactly the
        # observed case. Making these exclusive would force a choice that misdescribes it.
        for group in EXCLUSIVE_TAG_GROUPS:
            assert not {"bodies-rocking", "him-strong"} <= group
