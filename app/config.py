from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str
    jwt_secret: str
    jwt_expiry_hours: int = 24
    s3_jobs_bucket: str = "wanly-jobs"
    s3_loras_bucket: str = "wanly-loras"
    s3_faces_bucket: str = "wanly-faces"
    s3_images_bucket: str = "wanly-images"
    aws_region: str = "us-west-2"
    api_key: str = ""
    civitai_api_token: str = ""
    cors_origins: str = ""
    login_rate_limit: str = "5/minute"
    ollama_url: str = "http://2070.zero:11434"
    ollama_vision_model: str = "user-v4/joycaption-beta:latest"
    ollama_text_model: str = "dolphin-mistral:7b"
    ollama_vision_prompt: str = "Describe this image in explicit detail in under 100 words. Include all people, their positions, actions, body language, setting, lighting, and camera angle."
    ollama_text_prompt: str = "Here is a still image description of {prefix}:\n\n{description}\n\nWrite a single concise paragraph describing how this scene continues as a 5-second video clip. The camera remains completely static — no pans, zooms, or angle changes. Only describe body movements, rhythm, position changes, and physical reactions. Use explicit adult terminology for positions and actions. Keep it under 100 words."

    model_config = {"env_file": ".env"}


settings = Settings()
