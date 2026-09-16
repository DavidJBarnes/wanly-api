from typing import Literal, Optional

from pydantic import BaseModel, Field

from app.joycaption import (CAPTION_STYLES, DEFAULT_STYLE, MOTION_DEFAULT_STYLE,
                            MOTION_STYLE_PRESETS)

CaptionStyle = Literal["terse", "standard", "rich", "raw"]
MotionStyle = Literal["handheld", "amateur", "cinematic", "static", "none"]


class AppSettingsResponse(BaseModel):
    negative_prompt: str
    # How verbose the <SCENE> description should be (console#405). The presets exist because
    # the right length is a judgement about PROPORTION, not a universal answer: the caption
    # sits beside a ~100-word arc, and a description that outweighs it risks the model
    # holding the scene at the expense of the motion.
    # Defaulted, not required: these settings always have a value (see _DEFAULTS in
    # routes/app_settings.py), so a response can always be constructed.
    caption_style: CaptionStyle = DEFAULT_STYLE
    # Non-empty overrides the style entirely. The escape hatch: the presets encode what
    # tested well, but whoever is tuning prompts knows the material better than a default.
    caption_instruction: str = ""
    # Read-only, so the console can show what each style actually asks for rather than
    # making the user guess what "rich" means.
    caption_style_prompts: dict[str, str] = Field(default_factory=lambda: dict(CAPTION_STYLES))
    # The capture style of the motion half (#326). Same escape-hatch pattern.
    motion_style: MotionStyle = MOTION_DEFAULT_STYLE
    motion_instruction: str = ""
    motion_style_prompts: dict[str, str] = Field(
        default_factory=lambda: dict(MOTION_STYLE_PRESETS))


class AppSettingsUpdate(BaseModel):
    negative_prompt: Optional[str] = None
    caption_style: Optional[CaptionStyle] = None
    # Deliberately allows "" — that is how you CLEAR a custom instruction and fall back to
    # the style. exclude_none in the route means None is "leave alone" and "" is "clear",
    # which are different intents and must stay distinguishable.
    caption_instruction: Optional[str] = Field(default=None, max_length=2000)
    motion_style: Optional[MotionStyle] = None
    motion_instruction: Optional[str] = Field(default=None, max_length=2000)
