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

    model_config = {"env_file": ".env"}


settings = Settings()
