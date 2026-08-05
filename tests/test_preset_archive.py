"""Tests for archiving video settings presets.

Two days of identity experiments created ~36 throwaway presets against 8 real ones, burying the
recipes that matter. Deleting is not an option — jobs reference a preset by id, and that link is
the record of which config produced which result.

So archiving must hide a preset from the PICKER while leaving it fully resolvable by id. The
regression that would hurt is the inverse: an archived preset that 404s or returns NULL config
would silently rewrite the history of every job that used it.
"""

import inspect

import pytest

from app.models import VideoSettingsPreset
from app.routes import video_presets
from app.schemas.video_presets import VideoPresetResponse, VideoPresetUpdate


class TestModel:
    def test_archived_column_exists_and_is_not_nullable(self):
        col = VideoSettingsPreset.__table__.columns["archived"]
        assert col.nullable is False, "a NULL archived would be neither active nor archived"

    def test_defaults_to_active(self):
        """Existing rows and new presets must remain visible; archiving is opt-in."""
        col = VideoSettingsPreset.__table__.columns["archived"]
        assert col.default.arg is False
        assert col.server_default is not None, "existing rows need a backfill default"

    def test_column_is_on_the_preset_not_some_other_table(self):
        """It first landed on Segment, because a blind string replace hit the first
        `loras = mapped_column(JSON...)` in the file rather than the preset's."""
        from app.models import Segment
        assert "archived" in VideoSettingsPreset.__table__.columns
        assert "archived" not in Segment.__table__.columns


class TestListFiltering:
    def test_list_takes_an_include_archived_flag_defaulting_to_false(self):
        sig = inspect.signature(video_presets.list_video_presets)
        assert "include_archived" in sig.parameters
        assert sig.parameters["include_archived"].default.default is False

    def test_list_filters_on_the_archived_column(self):
        src = inspect.getsource(video_presets.list_video_presets)
        assert "include_archived" in src and "archived" in src
        assert "where" in src.lower(), "must actually filter, not just accept the flag"

    def test_only_the_list_route_filters_archived(self):
        """The whole point: historical jobs must keep resolving their preset. They do it via
        db.get() in the claim path, not through this router, so archiving can never break them
        -- but no OTHER route here may start filtering on archived either."""
        for fn in (video_presets.create_video_preset, video_presets.update_video_preset,
                   video_presets.delete_video_preset):
            assert "archived.is_(False)" not in inspect.getsource(fn), fn.__name__


class TestUpdate:
    def test_update_schema_exposes_archived(self):
        assert "archived" in VideoPresetUpdate.model_fields

    def test_archived_is_optional_so_partial_updates_do_not_unarchive(self):
        """A PATCH that only renames a preset must not silently flip it back to active."""
        assert VideoPresetUpdate.model_fields["archived"].default is None
        assert VideoPresetUpdate().archived is None

    def test_update_route_applies_archived(self):
        src = inspect.getsource(video_presets.update_video_preset)
        assert "body.archived" in src

    @pytest.mark.parametrize("value", [True, False])
    def test_round_trips_both_directions(self, value):
        """Unarchiving matters as much as archiving — that is what makes this reversible."""
        assert VideoPresetUpdate(archived=value).archived is value


class TestResponse:
    def test_response_exposes_archived_so_the_ui_can_show_state(self):
        assert "archived" in VideoPresetResponse.model_fields
