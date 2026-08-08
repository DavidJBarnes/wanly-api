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
    # Community 4090, no network volume — decided 2026-08-08 on measured cost.
    #
    # Per 720p segment: community 4090 $0.104, secure 4090 + volume $0.226, community 3090
    # $0.109. Community 4090 is the same price per segment as a 3090 while being ~1.6x faster
    # (1098s vs 1781s), and it drops the volume's $7/mo. Secure never breaks even: it is cheaper
    # only on boot (2 min vs 13), which one segment of runtime repays.
    #
    # The trade is that community cannot mount a network volume, so every pod re-downloads ~39GB
    # (~13 min, ~$0.07). Worth it when a pod stays up for several segments; not when launching
    # repeatedly for one.
    #
    # All three are settings rather than constants so switching back is config, not a deploy.
    # Empty volume id and empty datacenter mean "do not pin" — required for community, which has
    # neither.
    runpod_api_key: str = ""
    runpod_cloud_type: str = "COMMUNITY"
    runpod_network_volume_id: str = ""
    runpod_datacenter_id: str = ""
    runpod_gpu_type_id: str = "NVIDIA GeForce RTX 4090"
    # GPUs the launcher offers, comma separated. The 4090 is preferred but community 4090s are
    # frequently unplaceable — RunPod matches a host and then reports "this machine does not have
    # the resources", because the community fleet is largely partially committed. The 3090 is the
    # fallback that does place: slower per segment, but roughly 2/3 the price and actually
    # obtainable. Anything listed here must be verified against our image; the 5090 is
    # deliberately absent because the workflow does not run on it.
    runpod_gpu_type_ids: str = "NVIDIA GeForce RTX 4090,NVIDIA GeForce RTX 3090"
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
