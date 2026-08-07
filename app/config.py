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
    heartbeat_offline_seconds: int = 120

    # RunPod worker launching (wanly-console#288).
    #
    # The key lives here rather than in the browser: creating a pod needs a read/write RunPod
    # key, which can also terminate pods and volumes. There is no launch-only scope, so it must
    # never reach the client.
    #
    # Defaults are the measured-best configuration, not arbitrary. 4090 because 3090 secure
    # inventory is transient (present in one sample, gone the next) and could not be honoured.
    # EU-RO-1 because network volumes are region-locked and it had 4090 capacity in 10/10
    # samples, while every US storage-capable datacenter had 0-1/10.
    runpod_api_key: str = ""
    runpod_network_volume_id: str = ""
    runpod_datacenter_id: str = "EU-RO-1"
    runpod_gpu_type_id: str = "NVIDIA GeForce RTX 4090"
    runpod_image: str = "davidjbarnes/wanly-gpu-docker:latest"
    runpod_container_disk_gb: int = 30
    runpod_volume_mount_path: str = "/workspace"
    # What the launched worker is told to poll. Must be reachable FROM RunPod, so it cannot be
    # localhost even though this server is the thing being pointed at.
    runpod_worker_queue_url: str = "http://api.wanly22.com:8001"
    # Interim seam smoothing: crossfade (xfade) overlap between consecutive segments,
    # in seconds. 0 disables it (hard-cut concat, prior behavior). Superseded later by
    # VACE video-conditioned continuation.
    stitch_crossfade_seconds: float = 0.0

    model_config = {"env_file": ".env"}


settings = Settings()
