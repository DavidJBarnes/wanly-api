from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str
    jwt_secret: str
    jwt_expiry_hours: int = 24
    s3_jobs_bucket: str = "wanly-jobs"
    s3_faces_bucket: str = "wanly-faces"
    s3_images_bucket: str = "wanly-images"
    # Character LoRAs, so a worker can obtain one instead of only rendering on a box that
    # already has the file. Its own region: the bucket was created in us-east-1 while
    # everything else here is us-west-2, and a presigned URL signed for the wrong region
    # fails with SignatureDoesNotMatch — which reads like an auth problem and is not one.
    s3_loras_bucket: str = "ltx-loras"
    s3_loras_region: str = "us-east-1"
    aws_region: str = "us-west-2"
    api_key: str = ""
    civitai_api_token: str = ""
    # JoyCaption, for the <SCENE> placeholder (console#405). Runs on the 2070 rather than a
    # render box: the 3090 sits at ~23 of 24 GB while rendering LTX, and loading a vision
    # model beside that OOMs a segment ten minutes in.
    joycaption_url: str = "http://2070.zero:11434"
    joycaption_model: str = "joycaption:beta-one"
    # Deliberately tiny. sd.service (Automatic1111) shares that GPU and spikes several GB
    # generating SDXL, and a resident 5.5 GB JoyCaption would starve it. The two are never
    # meant to run at once, so the model should hold the card only while it is actually
    # working.
    #
    # This is cheap because a cold start is cheap. MEASURED on the 2070:
    #
    #   cold (not resident)   4.5 s total   2.9 s load, 0.5 s prompt, 0.9 s generate
    #   warm (resident)       1.2 s total   0.2 s load, 1.0 s generate
    #
    # So releasing VRAM after every caption costs ~2.9 s on the next one. 5s still coalesces
    # a burst of images captioned back to back, while giving SD the card back essentially as
    # soon as captioning stops.
    joycaption_keep_alive: str = "5s"
    # Generous next to a 4.5 s cold caption. It is here to stop a wedged or unreachable
    # captioner holding a request open, not to bound normal work.
    joycaption_timeout_s: int = 60
    # Automatic1111 on the same 2070, so a caption can ask it for the card back.
    #
    # The keep_alive above makes JoyCaption yield to A1111. Nothing made A1111 yield back,
    # and it holds its checkpoint whether or not it is generating — which is enough on its
    # own to abort a caption. See _yield_the_gpu in app/joycaption.py for the measurements.
    #
    # Empty disables it: anywhere the captioner does not share a card with A1111, there is
    # nothing to ask and nothing to reload.
    a1111_url: str = "http://2070.zero:7860"
    # Unloading is a few seconds of torch teardown. Short, because failing to free the card
    # only costs the caption, which is never fatal to a render.
    a1111_yield_timeout_s: int = 20
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
    # Writable container disk. Holds /jobs — every render's graph, keyframe and mp4, roughly
    # 30-60 MB a piece — plus ComfyUI's scratch. NOT the image, which RunPod accounts for
    # separately: a pod with a 30 GB container disk reports 30 GB free before anything runs.
    runpod_container_disk_gb: int = 40
    # Disk mounted at runpod_volume_mount_path. REQUIRED without a network volume, and it must
    # fit the LTX 2.3 model set, which download_models.sh stages into /workspace/models:
    #
    #   sulphur_dev_bf16.safetensors            43 GB
    #   gemma_3_12B_it_fp8_scaled.safetensors   13 GB
    #   ltx-2.3-spatial-upscaler-x2-1.1        950 MB
    #   sulphur_distill_lora_condsafe          632 MB
    #                                        ------- ~58 GB, plus character LoRAs at
    #                                                625 MB each, synced per claim.
    #
    # This was 60, sized for the WAN set (~37 GB) it no longer runs. A pod launched at 60 GB
    # cannot hold 58 GB of models plus LoRAs and dies partway through staging — which is what
    # happened on the first real pod, 2026-09-01.
    #
    # Setting volumeMountPath WITHOUT this allocates no volume at all, so /workspace silently
    # lands on the container disk. 0 disables (correct only when a network volume supplies the
    # mount instead) — and a network volume is the better shape for repeat launches, since it
    # pays the 43 GB checkpoint download once rather than per cold pod.
    runpod_volume_gb: int = 150
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
