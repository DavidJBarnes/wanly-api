import logging

import boto3

from app.config import settings

logger = logging.getLogger(__name__)

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = boto3.client("s3", region_name=settings.aws_region)
    return _client


def upload_bytes(data: bytes, key: str) -> str:
    """Upload bytes to S3. Returns the S3 URI."""
    client = _get_client()
    client.put_object(Bucket=settings.s3_bucket, Key=key, Body=data)
    uri = f"s3://{settings.s3_bucket}/{key}"
    logger.info("Uploaded %d bytes to %s", len(data), uri)
    return uri


def upload_file(path: str, key: str) -> str:
    """Upload a local file to S3 using multipart. Returns the S3 URI."""
    client = _get_client()
    client.upload_file(path, settings.s3_bucket, key)
    uri = f"s3://{settings.s3_bucket}/{key}"
    logger.info("Uploaded file %s to %s", path, uri)
    return uri


def download_bytes(uri: str) -> bytes:
    """Download bytes from an S3 URI (s3://bucket/key)."""
    parts = uri.replace("s3://", "").split("/", 1)
    bucket, key = parts[0], parts[1]
    client = _get_client()
    resp = client.get_object(Bucket=bucket, Key=key)
    return resp["Body"].read()


def delete_prefix(prefix: str) -> int:
    """Delete all objects under a prefix. Returns count of deleted objects."""
    client = _get_client()
    resp = client.list_objects_v2(Bucket=settings.s3_bucket, Prefix=prefix)
    objects = resp.get("Contents", [])
    if not objects:
        return 0
    delete_keys = [{"Key": obj["Key"]} for obj in objects]
    client.delete_objects(Bucket=settings.s3_bucket, Delete={"Objects": delete_keys})
    logger.info("Deleted %d objects under %s", len(delete_keys), prefix)
    return len(delete_keys)


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """Parse s3://bucket/key into (bucket, key)."""
    parts = uri.replace("s3://", "").split("/", 1)
    return parts[0], parts[1]
