"""Object storage for uploads + result JSON.

Uses Cloudflare R2 (S3-compatible, free tier, pairs with Pages) when R2 credentials
are configured; otherwise falls back to local disk so the backend runs end-to-end on
a laptop with no cloud setup. Same interface either way.
"""

from __future__ import annotations

import json
import os

import config


class _LocalStore:
    def __init__(self, root: str):
        self.root = root
        os.makedirs(os.path.join(root, "uploads"), exist_ok=True)
        os.makedirs(os.path.join(root, "results"), exist_ok=True)

    def put_upload(self, key: str, data: bytes) -> None:
        path = os.path.join(self.root, "uploads", key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)

    def get_upload(self, key: str) -> bytes:
        with open(os.path.join(self.root, "uploads", key), "rb") as f:
            return f.read()

    def put_result(self, key: str, result: dict) -> None:
        with open(os.path.join(self.root, "results", key + ".json"), "w") as f:
            json.dump(result, f)

    def get_result(self, key: str) -> dict | None:
        path = os.path.join(self.root, "results", key + ".json")
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return json.load(f)


class _R2Store:
    def __init__(self):
        import boto3
        self.bucket = config.R2_BUCKET
        self.s3 = boto3.client(
            "s3",
            endpoint_url=config.R2_ENDPOINT,
            aws_access_key_id=config.R2_ACCESS_KEY,
            aws_secret_access_key=config.R2_SECRET_KEY,
            region_name="auto",
        )

    def put_upload(self, key: str, data: bytes) -> None:
        self.s3.put_object(Bucket=self.bucket, Key=f"uploads/{key}", Body=data)

    def get_upload(self, key: str) -> bytes:
        obj = self.s3.get_object(Bucket=self.bucket, Key=f"uploads/{key}")
        return obj["Body"].read()

    def put_result(self, key: str, result: dict) -> None:
        self.s3.put_object(
            Bucket=self.bucket, Key=f"results/{key}.json",
            Body=json.dumps(result).encode(), ContentType="application/json",
        )

    def get_result(self, key: str) -> dict | None:
        try:
            obj = self.s3.get_object(Bucket=self.bucket, Key=f"results/{key}.json")
            return json.loads(obj["Body"].read())
        except self.s3.exceptions.NoSuchKey:
            return None


def get_store():
    if config.R2_ENDPOINT and config.R2_ACCESS_KEY:
        return _R2Store()
    return _LocalStore(config.LOCAL_STORAGE_DIR)
