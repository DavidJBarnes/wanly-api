from typing import Optional

from pydantic import BaseModel, model_validator


class PromptGenRequest(BaseModel):
    image_base64: Optional[str] = None
    image_s3_uri: Optional[str] = None
    prompt_prefix: Optional[str] = None

    @model_validator(mode="after")
    def exactly_one_image_source(self):
        has_base64 = bool(self.image_base64)
        has_s3 = bool(self.image_s3_uri)
        if not has_base64 and not has_s3:
            raise ValueError("Either image_base64 or image_s3_uri is required")
        if has_base64 and has_s3:
            raise ValueError("Provide only one of image_base64 or image_s3_uri")
        return self


class PromptGenResponse(BaseModel):
    prompt: str
    description: str


class PromptTemplatesResponse(BaseModel):
    vision_prompt: str
    text_prompt: str
    vision_model: str
    text_model: str


class PromptTemplatesUpdate(BaseModel):
    vision_prompt: Optional[str] = None
    text_prompt: Optional[str] = None
    vision_model: Optional[str] = None
    text_model: Optional[str] = None
