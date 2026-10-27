from typing import Annotated, Literal, Optional

from pydantic import BaseModel, Field

from app.joycaption import CAPTION_STYLES, DEFAULT_STYLE, PROMPT_MAX_CHARS

CaptionStyle = Literal["terse", "standard", "rich", "raw"]
QueuePosition = Literal["top", "bottom"]

#: A caption instruction as the editor sends it. No placeholders: the static half has
#: nothing to fill in, so braces in it are just text.
CaptionInstruction = Annotated[str, Field(max_length=PROMPT_MAX_CHARS)]


class AppSettingsResponse(BaseModel):
    negative_prompt: str
    # How verbose the <SCENE> description should be (console#405). The presets exist because
    # the right length is a judgement about PROPORTION, not a universal answer: the caption
    # sits beside a ~100-word arc, and a description that outweighs it risks the model
    # holding the scene at the expense of the action.
    # Defaulted, not required: these settings always have a value (see _DEFAULTS in
    # routes/app_settings.py), so a response can always be constructed.
    caption_style: CaptionStyle = DEFAULT_STYLE
    # Non-empty overrides the style entirely. The escape hatch: the presets encode what
    # tested well, but whoever is tuning prompts knows the material better than a default.
    caption_instruction: str = ""
    # Read-only: the DEFAULT text of every style. The console pre-fills its editor with it
    # and resets to it (console#555), so it never restates prompt text of its own.
    caption_style_prompts: dict[str, str] = Field(default_factory=lambda: dict(CAPTION_STYLES))
    # (The motion_* settings went with the motion caption half, wanly-api#481.)
    # The cap the editor is held to, so the console counts against the real number.
    prompt_max_length: int = PROMPT_MAX_CHARS
    # Where a newly queued segment lands (#471). "top": it keeps its job's place, so a job
    # under way runs ahead of jobs queued after it. "bottom": its job moves behind the rest.
    segment_queue_position: QueuePosition = "top"


class AppSettingsUpdate(BaseModel):
    negative_prompt: Optional[str] = None
    caption_style: Optional[CaptionStyle] = None
    # Deliberately allows "" — that is how you CLEAR a custom instruction and fall back to
    # the style. exclude_none in the route means None is "leave alone" and "" is "clear",
    # which are different intents and must stay distinguishable.
    caption_instruction: Optional[CaptionInstruction] = None
    # An older console still sending motion_style / motion_instruction is not refused: unknown
    # fields are ignored (pydantic's default), and nothing reads those keys any more.
    segment_queue_position: Optional[QueuePosition] = None
