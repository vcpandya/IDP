"""AWS S3 connector — composite (access key + secret + bucket + region)."""
from __future__ import annotations

import asyncio

from idpkit.connectors.base import (
    Connector, ConnectorAuthError, ConnectorAuthType, ConnectorError,
    ConnectorField, ConnectorTool,
)


def _client(creds: dict):
    try:
        import boto3  # type: ignore
    except ImportError as exc:
        raise ConnectorError(
            "boto3 is not installed in this environment; install it to use the S3 connector."
        ) from exc
    return boto3.client(
        "s3",
        aws_access_key_id=creds["access_key_id"],
        aws_secret_access_key=creds["secret_access_key"],
        region_name=creds.get("region") or "us-east-1",
    )


async def health_check(creds: dict) -> tuple[bool, str]:
    bucket = creds.get("bucket", "")

    def _check():
        c = _client(creds)
        try:
            c.head_bucket(Bucket=bucket)
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            if "403" in msg or "InvalidAccessKey" in msg or "SignatureDoesNotMatch" in msg:
                raise ConnectorAuthError(f"S3 auth failed: {msg[:200]}")
            raise ConnectorError(f"S3 head_bucket failed: {msg[:200]}")

    await asyncio.to_thread(_check)
    return True, f"s3://{bucket}"


async def _list_objects(args: dict, creds: dict) -> dict:
    prefix = args.get("prefix", "")
    max_keys = min(int(args.get("max_keys", 50)), 1000)

    def _do():
        c = _client(creds)
        return c.list_objects_v2(Bucket=creds["bucket"], Prefix=prefix, MaxKeys=max_keys)

    resp = await asyncio.to_thread(_do)
    contents = resp.get("Contents", []) if isinstance(resp, dict) else []
    return {"objects": [
        {"key": o.get("Key"), "size": o.get("Size"), "last_modified": str(o.get("LastModified"))}
        for o in contents
    ]}


async def _presign_url(args: dict, creds: dict) -> dict:
    key = args.get("key", "")
    expires = min(int(args.get("expires_in", 3600)), 7 * 24 * 3600)
    if not key:
        return {"error": "key is required"}

    def _do():
        c = _client(creds)
        return c.generate_presigned_url(
            "get_object",
            Params={"Bucket": creds["bucket"], "Key": key},
            ExpiresIn=expires,
        )

    url = await asyncio.to_thread(_do)
    return {"url": url, "expires_in": expires}


CONNECTOR = Connector(
    id="s3",
    display_name="AWS S3",
    description="List objects and generate presigned URLs in an S3 bucket.",
    icon="fa-brands fa-aws",
    auth_type=ConnectorAuthType.COMPOSITE,
    fields=[
        ConnectorField(key="access_key_id", label="Access Key ID", type="text", placeholder="AKIA..."),
        ConnectorField(key="secret_access_key", label="Secret Access Key", type="password"),
        ConnectorField(key="bucket", label="Bucket Name", type="text", placeholder="my-bucket"),
        ConnectorField(
            key="region", label="Region", type="text", placeholder="us-east-1", required=False,
        ),
    ],
    tools=[
        ConnectorTool(
            name="s3_list_objects",
            description="List objects in the configured S3 bucket under an optional prefix.",
            parameters={
                "type": "object",
                "properties": {
                    "prefix": {"type": "string", "default": ""},
                    "max_keys": {"type": "integer", "default": 50, "minimum": 1, "maximum": 1000},
                },
            },
            executor=_list_objects,
        ),
        ConnectorTool(
            name="s3_presigned_url",
            description="Generate a temporary presigned download URL for an S3 object.",
            parameters={
                "type": "object",
                "properties": {
                    "key": {"type": "string"},
                    "expires_in": {"type": "integer", "default": 3600, "minimum": 60, "maximum": 604800},
                },
                "required": ["key"],
            },
            executor=_presign_url,
        ),
    ],
    docs_url="https://docs.aws.amazon.com/AmazonS3/latest/userguide/Welcome.html",
    health_check=health_check,
)
