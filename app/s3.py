import logging
import mimetypes

import boto3

from app.config import settings

logger = logging.getLogger(__name__)

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = boto3.client("s3", region_name=settings.aws_region)
    return _client


_IMMUTABLE_CACHE_CONTROL = "public, max-age=86400, immutable"


def _content_type_for(key: str) -> str:
    guess, _ = mimetypes.guess_type(key)
    return guess or "binary/octet-stream"


def upload_bytes(data: bytes, key: str, bucket: str) -> str:
    """Upload bytes to S3. Returns the S3 URI."""
    client = _get_client()
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=data,
        CacheControl=_IMMUTABLE_CACHE_CONTROL,
        ContentType=_content_type_for(key),
    )
    uri = f"s3://{bucket}/{key}"
    logger.info("Uploaded %d bytes to %s", len(data), uri)
    return uri


def upload_file(path: str, key: str, bucket: str) -> str:
    """Upload a local file to S3 using multipart. Returns the S3 URI."""
    client = _get_client()
    client.upload_file(
        path,
        bucket,
        key,
        ExtraArgs={
            "CacheControl": _IMMUTABLE_CACHE_CONTROL,
            "ContentType": _content_type_for(key),
        },
    )
    uri = f"s3://{bucket}/{key}"
    logger.info("Uploaded file %s to %s", path, uri)
    return uri


def download_bytes(uri: str) -> bytes:
    """Download bytes from an S3 URI (s3://bucket/key)."""
    parts = uri.replace("s3://", "").split("/", 1)
    bucket, key = parts[0], parts[1]
    client = _get_client()
    resp = client.get_object(Bucket=bucket, Key=key)
    return resp["Body"].read()


def delete_prefix(prefix: str, bucket: str) -> int:
    """Delete all objects under a prefix (paginated). Returns count of deleted objects."""
    client = _get_client()
    total_deleted = 0
    params: dict = {"Bucket": bucket, "Prefix": prefix}
    while True:
        resp = client.list_objects_v2(**params)
        objects = resp.get("Contents", [])
        if not objects:
            break
        delete_keys = [{"Key": obj["Key"]} for obj in objects]
        client.delete_objects(Bucket=bucket, Delete={"Objects": delete_keys})
        total_deleted += len(delete_keys)
        if not resp.get("IsTruncated"):
            break
        params["ContinuationToken"] = resp["NextContinuationToken"]
    if total_deleted:
        logger.info("Deleted %d objects under %s/%s", total_deleted, bucket, prefix)
    return total_deleted


def delete_prefix_except(prefix: str, bucket: str, except_uris: set[str]) -> int:
    """Delete objects under a prefix, skipping any whose s3:// URI is in except_uris.

    Used when a job-prefix cleanup needs to avoid deleting shared content still
    referenced by other jobs (e.g. a legacy deduped starting image).
    """
    client = _get_client()
    prefix_uri = f"s3://{bucket}/"
    except_keys = {
        uri[len(prefix_uri):] for uri in except_uris if uri.startswith(prefix_uri)
    }
    total_deleted = 0
    params: dict = {"Bucket": bucket, "Prefix": prefix}
    while True:
        resp = client.list_objects_v2(**params)
        objects = resp.get("Contents", [])
        to_delete = [{"Key": o["Key"]} for o in objects if o["Key"] not in except_keys]
        if to_delete:
            client.delete_objects(Bucket=bucket, Delete={"Objects": to_delete})
            total_deleted += len(to_delete)
        if not resp.get("IsTruncated"):
            break
        params["ContinuationToken"] = resp["NextContinuationToken"]
    if total_deleted:
        logger.info("Deleted %d objects under %s/%s (kept %d shared)",
                    total_deleted, bucket, prefix, len(except_keys))
    return total_deleted


def delete_object(uri: str) -> None:
    """Delete a single object by S3 URI."""
    bucket, key = parse_s3_uri(uri)
    client = _get_client()
    client.delete_object(Bucket=bucket, Key=key)
    logger.info("Deleted %s/%s", bucket, key)


def download_file(uri: str, local_path: str) -> None:
    """Download an S3 object to a local file (streams to disk)."""
    bucket, key = parse_s3_uri(uri)
    client = _get_client()
    client.download_file(bucket, key, local_path)
    logger.info("Downloaded %s to %s", uri, local_path)


def list_common_prefixes(bucket: str, prefix: str = "", delimiter: str = "/") -> list[str]:
    """List virtual folders. Returns prefixes like ['2026-02-27/']."""
    client = _get_client()
    prefixes: list[str] = []
    params: dict = {"Bucket": bucket, "Prefix": prefix, "Delimiter": delimiter}
    while True:
        resp = client.list_objects_v2(**params)
        for cp in resp.get("CommonPrefixes", []):
            prefixes.append(cp["Prefix"])
        if not resp.get("IsTruncated"):
            break
        params["ContinuationToken"] = resp["NextContinuationToken"]
    return prefixes


def list_objects(bucket: str, prefix: str) -> list[dict]:
    """List all objects under a prefix. Returns [{Key, Size, LastModified}]."""
    client = _get_client()
    objects: list[dict] = []
    params: dict = {"Bucket": bucket, "Prefix": prefix}
    while True:
        resp = client.list_objects_v2(**params)
        for obj in resp.get("Contents", []):
            objects.append({
                "Key": obj["Key"],
                "Size": obj["Size"],
                "LastModified": obj["LastModified"].isoformat(),
            })
        if not resp.get("IsTruncated"):
            break
        params["ContinuationToken"] = resp["NextContinuationToken"]
    return objects


def get_folder_info(bucket: str, prefix: str) -> dict | None:
    """Return thumbnail key and creation date for a folder prefix.

    Uses the .folder marker's LastModified for creation time if present,
    otherwise falls back to the first non-marker object.
    """
    client = _get_client()
    resp = client.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=20)
    contents = resp.get("Contents", [])
    if not contents:
        return None

    marker = None
    first_image = None
    for obj in contents:
        if obj["Key"].endswith("/.folder"):
            marker = obj
        elif first_image is None:
            first_image = obj
        if marker and first_image:
            break

    key = first_image["Key"] if first_image else None
    created_at_obj = marker if marker else first_image
    created_at = created_at_obj["LastModified"].isoformat() if created_at_obj else None
    return {"key": key, "created_at": created_at}


def move_object(bucket: str, src_key: str, dst_key: str) -> None:
    """Copy an object within the same bucket then delete the original."""
    client = _get_client()
    client.copy_object(
        Bucket=bucket,
        CopySource={"Bucket": bucket, "Key": src_key},
        Key=dst_key,
    )
    client.delete_object(Bucket=bucket, Key=src_key)
    logger.info("Moved %s/%s → %s/%s", bucket, src_key, bucket, dst_key)


def generate_presigned_url(uri: str, expires: int = 21600) -> str:
    """Generate a presigned GET URL for an S3 URI. Default expiry: 6 hours.

    The /api/files redirect caches for <6h so the browser never replays a
    cached redirect that points to an expired signature.
    """
    bucket, key = parse_s3_uri(uri)
    client = _get_client()
    return client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=expires,
    )


def head_object(uri: str) -> dict | None:
    """Return {Key, Size, LastModified} for a single S3 object, or None."""
    bucket, key = parse_s3_uri(uri)
    client = _get_client()
    try:
        resp = client.head_object(Bucket=bucket, Key=key)
        return {
            "Key": key,
            "Size": resp["ContentLength"],
            "LastModified": resp["LastModified"].isoformat(),
        }
    except Exception:
        return None


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """Parse s3://bucket/key into (bucket, key)."""
    parts = uri.replace("s3://", "").split("/", 1)
    return parts[0], parts[1]
