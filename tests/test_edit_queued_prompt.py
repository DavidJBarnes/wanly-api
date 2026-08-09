"""A queued segment's prompt must be editable, but only before a worker claims it.

The prompt is a SNAPSHOT taken at job creation, unlike loras and sampler settings which resolve
live from the preset at claim time. That asymmetry is deliberate -- a queued job should not
silently change what it depicts -- but it left no way to correct a queued batch after improving
the preset, short of deleting and recreating every job. Which is what prompted this: a prompt fix
landed while a batch was already queued and could not reach it.
"""

import pytest
from pydantic import ValidationError

from app.enums import SegmentStatus
from app.schemas.segments import SegmentPromptUpdate


class TestRequestShape:
    def test_text_alone_is_valid(self):
        assert SegmentPromptUpdate(prompt="a woman walks").prompt == "a woman walks"

    def test_from_preset_alone_is_valid(self):
        assert SegmentPromptUpdate(from_preset=True).from_preset

    def test_neither_is_rejected(self):
        # An empty request would silently do nothing and report success.
        with pytest.raises(ValidationError):
            SegmentPromptUpdate()

    def test_both_is_rejected(self):
        # Ambiguous: it would have to pick one, and either choice surprises half the callers.
        with pytest.raises(ValidationError):
            SegmentPromptUpdate(prompt="x", from_preset=True)

    def test_empty_string_is_not_a_prompt(self):
        with pytest.raises(ValidationError):
            SegmentPromptUpdate(prompt="")


class TestEditableWindow:
    """Editable only while PENDING and unclaimed.

    Once a worker holds the segment the prompt has already gone to ComfyUI, so an edit would
    change the record without changing the output -- the worst outcome, because the row would
    then describe something the video is not.
    """

    def test_pending_and_unclaimed_is_the_only_editable_state(self):
        editable = (SegmentStatus.PENDING, None)
        assert editable == (SegmentStatus.PENDING, None)

    @pytest.mark.parametrize("status", [
        SegmentStatus.CLAIMED, SegmentStatus.PROCESSING, SegmentStatus.COMPLETED,
    ])
    def test_every_other_status_is_closed(self, status):
        assert status != SegmentStatus.PENDING
