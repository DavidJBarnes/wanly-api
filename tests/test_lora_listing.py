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
