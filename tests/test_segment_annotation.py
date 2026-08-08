"""Tests for segment annotation (human observations).

The point of these fields is that the automated metrics cannot rank quality -- on the one
controlled pair where both exist, the expression score ranked them backwards, because a gaping
mouth is the largest landmark excursion a face can make and the metric rewards it (daemon#123).

So the valuable property is that tags GROUP. A vocabulary that admits near-misses produces
"mouth-void" and "mouth void" as separate labels and destroys the only thing they are for.
"""

import pytest

from app.schemas.segments import OBSERVATION_TAGS, SegmentAnnotation


class TestVocabulary:
    def test_covers_the_artifacts_actually_observed(self):
        # Named from real reports, not invented: the open-mouth failure was described as
        # "blank/black space mouth" with "messy teeth and lips".
        for tag in ("mouth-void", "teeth-mush", "frozen-face"):
            assert tag in OBSERVATION_TAGS

    def test_carries_positive_labels_too(self):
        # Rating is one all-in number, so per-axis signal has to come from somewhere. Positive
        # tags are what let "the motion was good but the mouth was a void" survive as data.
        for tag in ("good-motion", "good-expression", "good-detail"):
            assert tag in OBSERVATION_TAGS

    def test_no_duplicates(self):
        assert len(OBSERVATION_TAGS) == len(set(OBSERVATION_TAGS))


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
        chosen = {"good-motion", "mouth-void"}
        ordered = [t for t in OBSERVATION_TAGS if t in chosen]
        reversed_input = {"mouth-void", "good-motion"}
        assert ordered == [t for t in OBSERVATION_TAGS if t in reversed_input]
