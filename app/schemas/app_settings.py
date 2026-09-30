from typing import Annotated, Literal, Optional

from pydantic import AfterValidator, BaseModel, Field

from app.joycaption import (CAPTION_STYLES, DEFAULT_STYLE, MOTION_DEFAULT_STYLE,
                            MOTION_INSTRUCTIONS, MOTION_STYLE_PRESETS, PROMPT_MAX_CHARS,
                            strip_motion_tags)

CaptionStyle = Literal["terse", "standard", "rich", "raw"]
MotionStyle = Literal["handheld", "amateur", "cinematic", "static", "none"]

#: A caption instruction as the editor sends it. No placeholders: the static half has
#: nothing to fill in, so braces in it are just text.
CaptionInstruction = Annotated[str, Field(max_length=PROMPT_MAX_CHARS)]

#: Motion INSTRUCTIONS as the editor sends them (console#573): plain text, which the API
#: wraps in the style sentence and the grounding. A #555 template -- from a console tab
#: opened before this deployed -- is not refused but read the way a saved one is migrated,
#: its tags and grounding sections stripped, so it cannot be stored as literal plumbing.
#: Every input that carries motion instructions uses this type, so the rule cannot be
#: skipped by picking a different endpoint.
MotionInstructions = Annotated[str, Field(max_length=PROMPT_MAX_CHARS),
                               AfterValidator(strip_motion_tags)]


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
    # Read-only: the DEFAULT text of every style. The console pre-fills its editor with it
    # and resets to it (console#555), so it never restates prompt text of its own.
    caption_style_prompts: dict[str, str] = Field(default_factory=lambda: dict(CAPTION_STYLES))
    # The capture style of the motion half (#326). Same escape-hatch pattern.
    motion_style: MotionStyle = MOTION_DEFAULT_STYLE
    # The saved motion INSTRUCTIONS (console#573); "" means motion_instruction_default. A
    # saved #555 template is returned already migrated to its instructions.
    motion_instruction: str = ""
    # A pre-#555 whole-prompt override still in force: sent exactly as written, with no
    # style sentence and no grounding. "" when there is none (the usual case). Saving
    # motion_instruction replaces it.
    motion_legacy_prompt: str = ""
    motion_style_prompts: dict[str, str] = Field(
        default_factory=lambda: dict(MOTION_STYLE_PRESETS))
    # Read-only: the instructions "" stands for.
    motion_instruction_default: str = MOTION_INSTRUCTIONS
    # The cap both editors are held to, so the console counts against the real number.
    prompt_max_length: int = PROMPT_MAX_CHARS


class AppSettingsUpdate(BaseModel):
    negative_prompt: Optional[str] = None
    caption_style: Optional[CaptionStyle] = None
    # Deliberately allows "" — that is how you CLEAR a custom instruction and fall back to
    # the style. exclude_none in the route means None is "leave alone" and "" is "clear",
    # which are different intents and must stay distinguishable.
    caption_instruction: Optional[CaptionInstruction] = None
    motion_style: Optional[MotionStyle] = None
    motion_instruction: Optional[MotionInstructions] = None
