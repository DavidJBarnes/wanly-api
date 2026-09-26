from pydantic import AliasChoices, Field
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
    # image-description, for the <SCENE> placeholder (console#405) and dataset tagging. Named
    # for the capability (wanly-gpu-docker#83). As of #326 it is Qwen2.5-VL served by ollama
    # inside the wanly-services container on the 2070, chosen because one model does both
    # halves of a video prompt: the static scene AND the motion paragraph. The old
    # JOYCAPTION_* aliases still work.
    #
    # 2070-only, decided in #326: the 3090's captioner shares its card with the render stack,
    # and the single model-name setting cannot serve two models; the 2070's card is shared
    # with nothing that renders LTX. The 3090's joycaption stays installed, unused.
    image_description_url: str = Field(
        "http://2070.zero:11434",
        validation_alias=AliasChoices("image_description_url", "joycaption_url"))
    # A second captioner, used while the box above is rendering. The 2070 is the primary
    # now and nothing it shares its card with renders, so this is empty by default — the
    # mechanism stays for anyone who points the primary back at the 3090.
    image_description_fallback_url: str = ""
    image_description_model: str = Field(
        "qwen2.5vl:7b-q4_K_M",
        validation_alias=AliasChoices("image_description_model", "joycaption_model"))
    # Long, on purpose (#326). A cold qwen2.5vl load on the 2070 is ~3 minutes and a warm
    # CPU caption is 15-50 s, so 5s would make every caption pay for the load. Production
    # currently pins 5s by env because the deployment is back on joycaption:beta-one, whose
    # GPU load is ~3 s and which must yield the 8 GB card to A1111 (sd.service) promptly.
    # The ComfyUI dev install on this box does not fit beside the 15m residency — see the
    # runtime record in wanly-api#326.
    image_description_keep_alive: str = Field(
        "15m", validation_alias=AliasChoices("image_description_keep_alive", "joycaption_keep_alive"))
    # A cold qwen2.5vl caption (load included) measured 185-265 s on the 2070; a grounded
    # warm motion caption up to ~50 s. 180 — the old JoyCaption number — timed the first
    # cold call out. 600 leaves room for the load without letting a wedged captioner hold
    # a request open forever.
    #: THE CONTEXT WINDOW FOR A CAPTION, and the single biggest thing about caption speed.
    #:
    #: Sending no options lets ollama size the context from VRAM: on the 3090 it picks
    #: 32768, which needs 8 GB of KV cache, which does not fit beside a 20 GB model on a
    #: 24 GB card -- so 11 of 65 layers run on the CPU and every generated token crosses
    #: them. Measured, same image, same prompt, warm both times:
    #:
    #:     default (32768)   7.6s   7.8 tok/s   11 layers on CPU
    #:     4096              1.5s  37.3 tok/s   all layers on GPU
    #:
    #: A caption does not want 32k. The static prompt measured 451 tokens INCLUDING the
    #: image, and the motion prompt adds the scene paragraph -- so 4096 is roughly eight
    #: times what the job needs, and every byte above that is paid for in offloaded layers.
    image_description_num_ctx: int = Field(
        4096, validation_alias=AliasChoices("image_description_num_ctx",
                                            "joycaption_num_ctx"))
    image_description_timeout_s: int = Field(
        600, validation_alias=AliasChoices("image_description_timeout_s", "joycaption_timeout_s"))
    # The motion half of a description (#326) is an env kill-switch, and it exists because
    # the 2070's 8 GB cannot GPU-run qwen2.5vl:7b: measured with the card empty at 2048 ctx,
    # ollama offloads 1/29 layers and the model runs 100% CPU at 0.3-5 tok/s — a caption
    # takes minutes, and ~13 GB of system RAM the box does not have while A1111 holds a
    # checkpoint. With a JoyCaption-class model on the card the motion prompt is answered by
    # a model that cannot do it, producing plausible junk persisted as authoritative. Flip
    # this back to true when the captioner model can actually do the motion half.
    motion_caption_enabled: bool = True
    # Automatic1111 on the same 2070, so a caption can ask it for the card back.
    #
    # The keep_alive above makes JoyCaption yield to A1111. Nothing made A1111 yield back,
    # and it holds its checkpoint whether or not it is generating — which is enough on its
    # own to abort a caption. See _yield_the_gpu in app/joycaption.py for the measurements.
    #
    # Empty disables it: anywhere the captioner does not share a card with A1111, there is
    # nothing to ask and nothing to reload.
    # face-crop, in wanly-services. Called inline like JoyCaption is, and for the same reason:
    # detection on a handful of images is seconds of CPU, not a job worth queueing.
    #
    # Empty disables cropping from the console, which is the honest state anywhere the service
    # is not deployed — the button says so rather than timing out.
    face_crop_url: str = "http://3090.zero:8084"
    #: The port a worker's container serves its control API on (/health, /mode). One number
    #: for every box on purpose: it is set by run-worker.sh's CONTROL_PORT, which defaults to
    #: 8081 everywhere, and a per-worker value would have to be reported at registration and
    #: kept correct -- a lot of machinery for a constant.
    worker_control_port: int = 8081
    # Generous. It is per REQUEST, and a request carries a whole dataset — 50 images at a
    # second or two each on CPU.
    face_crop_timeout_s: int = 300
    # buffalo_l's same-person floor. Below this against a picked anchor is a different person;
    # it is shown as a line on the scores rather than used to delete anything, because the two
    # people who got into this project's training sets got there past a human eye, not past a
    # number nobody was shown.
    face_cos_floor: float = 0.4
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
    # Passed to pods as HF_TOKEN so model staging is authenticated (#260). huggingface_hub
    # reads it from the environment on its own, so download_models.sh and the image are
    # untouched -- the whole job is getting the variable into the pod.
    #
    # Empty means not configured, and the key is then OMITTED rather than sent blank: an
    # empty HF_TOKEN is worse than none, because huggingface_hub would try to authenticate
    # with it. Same shape as runpod_api_key above.
    #
    # Anonymous staging works today -- the repos are public and measured 14 MB/s -- so this
    # is about the cases where it stops working: HF's anonymous limits are per-IP and pods
    # share datacenter egress, and a gated repo 401s outright rather than warning.
    hf_token: str = ""
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
