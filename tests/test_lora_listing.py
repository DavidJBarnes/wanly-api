"""GET /loras — what a worker diffs against.

Workers hold no AWS credentials on purpose, so they cannot list the bucket themselves. This
endpoint is the half that tells them what exists; they already download through GET /files.

The rule that matters is what a worker COMPARES. Name is not enough: a retrained LoRA
republished under the same name would never be picked up, and the worker would render old
weights while the console showed the new character — wrong output rather than a failure.
"""
from app.config import settings
from app.s3 import _client_for_bucket


def test_the_loras_bucket_is_signed_in_its_own_region():
    """ltx-loras is us-east-1 while everything else is us-west-2.

    A presigned URL signed by a client in the wrong region is accepted at signing and
    rejected at use with SignatureDoesNotMatch, which reads like an auth bug and is not one.
    """
    loras = _client_for_bucket(settings.s3_loras_bucket)
    jobs = _client_for_bucket(settings.s3_jobs_bucket)
    assert loras.meta.region_name == settings.s3_loras_region == "us-east-1"
    assert jobs.meta.region_name == settings.aws_region == "us-west-2"
    assert loras.meta.region_name != jobs.meta.region_name


def test_a_multipart_etag_is_flagged_rather_than_trusted():
    """An etag ending in -<n> is not the md5 of the object.

    Treating it as a hash means the local file never matches and the worker re-downloads it
    on every sync, forever. Flagged so the caller can fall back to size and say so.
    """
    from app.s3 import list_bucket
    import app.s3 as s3mod

    class FakeClient:
        meta = type("m", (), {"region_name": "us-east-1"})()

        def list_objects_v2(self, **kw):
            return {"Contents": [
                {"Key": "single.safetensors", "Size": 10, "ETag": '"abc123"'},
                {"Key": "big.safetensors", "Size": 20, "ETag": '"def456-7"'},
            ], "IsTruncated": False}

    orig = s3mod._client_for_bucket
    s3mod._client_for_bucket = lambda b: FakeClient()
    try:
        out = {o["name"]: o for o in list_bucket("ltx-loras")}
    finally:
        s3mod._client_for_bucket = orig

    assert out["single.safetensors"]["multipart"] is False
    assert out["single.safetensors"]["etag"] == "abc123"      # quotes stripped
    assert out["big.safetensors"]["multipart"] is True        # -7 suffix


def test_loras_is_reachable_by_both_the_worker_and_the_console():
    """The dropdown and the sync list are the same endpoint, so both callers must get in.

    verify_api_key_or_token — the near-miss — accepts X-API-Key or a ?token= QUERY PARAM,
    which exists for <img src> media loads that cannot set headers. The console sends
    Authorization: Bearer, so that variant would have 401'd the dropdown while the worker
    kept working: broken for exactly one of the two callers, and only in the browser.
    """
    from app.auth import verify_api_key_or_bearer
    from app.main import app

    route = next(r for r in app.routes if getattr(r, "path", None) == "/loras")
    deps = [d.call for d in route.dependant.dependencies]
    assert verify_api_key_or_bearer in deps, (
        "GET /loras must accept a console Bearer token as well as a worker API key"
    )


async def test_a_prefixed_key_reports_its_kind_and_a_bare_name():
    """`name` must stay the BASENAME even though the key is prefixed.

    A ComfyUI LoraLoader takes `k3lly2026_v2.safetensors` and ltx_characters.char_lora
    stores the same. Neither knows the bucket has shelves. If `name` carried the prefix,
    the console would save "character/k3lly2026_v2" into char_lora and every render with
    that character would fail to find its LoRA.
    """
    import app.routes.ltx_recipes as mod

    fake = [
        {"name": "character/k3lly2026_v2.safetensors", "size": 10, "etag": "a", "multipart": False},
        {"name": "content/sfbehind_LTX2_3_v0_1.safetensors", "size": 20, "etag": "b", "multipart": False},
        {"name": "stray.safetensors", "size": 30, "etag": "c", "multipart": False},
        {"name": "notes.txt", "size": 1, "etag": "d", "multipart": False},
    ]
    orig = mod.list_bucket
    mod.list_bucket = lambda bucket: fake
    try:
        out = {o["name"]: o for o in await mod.list_available_loras()}
    finally:
        mod.list_bucket = orig

    assert out["k3lly2026_v2.safetensors"]["kind"] == "character"
    assert out["k3lly2026_v2.safetensors"]["key"] == "character/k3lly2026_v2.safetensors"
    assert out["k3lly2026_v2.safetensors"]["uri"].endswith("/character/k3lly2026_v2.safetensors")
    assert out["sfbehind_LTX2_3_v0_1.safetensors"]["kind"] == "content"

    # A LoRA at the root is surfaced as "unfiled", not hidden: a file nobody can see is
    # worse than one that is merely mis-shelved.
    assert out["stray.safetensors"]["kind"] == "unfiled"

    # Non-safetensors are not LoRAs.
    assert "notes.txt" not in out
