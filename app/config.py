from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str
    jwt_secret: str
    jwt_expiry_hours: int = 24
    s3_bucket: str = "wanly"
    aws_region: str = "us-west-2"
    civitai_api_token: str = ""

    model_config = {"env_file": ".env"}


settings = Settings()
