"""Uploaded issues and their page previews in Cloudflare R2 (S3 API, region "auto").

The browser uploads the PDF straight to R2 with a presigned PUT URL (issues are 15–60 MB, so they don't pass
through the app); the worker downloads it to a temp file. Page previews for the review page are JPEGs next to it.

  R2_ACCOUNT_ID, R2_ENDPOINT (optional: defaults to https://<account>.r2.cloudflarestorage.com), R2_BUCKET,
  R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY

Keys:  <kb_id>/<yyyy-mm or upload date>/<sha256>.pdf          the original
       <kb_id>/<yyyy-mm or upload date>/<sha256>/p001.jpg     page previews
"""
from __future__ import annotations

import hashlib
import os
import re
from datetime import date

MAX_UPLOAD_BYTES = 250 * 1024 * 1024
_MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]


class StorageNotConfigured(RuntimeError):
    pass


def configured() -> bool:
    return bool(os.environ.get("R2_BUCKET") and os.environ.get("R2_ACCESS_KEY_ID")
                and os.environ.get("R2_SECRET_ACCESS_KEY")
                and (os.environ.get("R2_ENDPOINT") or os.environ.get("R2_ACCOUNT_ID")))


def issue_folder(filename: str, today: date | None = None) -> str:
    """"yyyy-mm" if the file name says which issue it is ("Sunbeam 2023-01 January.pdf", "Sunbeam August 2024
    Issue.pdf"), else the upload date (yyyy-mm-dd)."""
    name = (filename or "").lower()
    m = re.search(r"\b(19|20)(\d\d)[-_ .](0[1-9]|1[0-2])\b", name)
    if m:
        return f"{m.group(1)}{m.group(2)}-{m.group(3)}"
    year = re.search(r"(19|20)\d\d", name)
    month = next((i for i, mon in enumerate(_MONTHS, 1) if re.search(rf"(?<![a-z]){mon}", name)), None)
    if year and month:
        return f"{year.group(0)}-{month:02d}"
    return (today or date.today()).isoformat()


def object_key(kb_id: str, filename: str, sha256: str, today: date | None = None) -> str:
    return f"{kb_id}/{issue_folder(filename, today)}/{sha256}.pdf"


def preview_key(pdf_key: str, page: int) -> str:
    return f"{pdf_key.removesuffix('.pdf')}/p{page:03d}.jpg"


class Storage:
    def __init__(self, client=None, bucket: str | None = None):
        if client is None:
            if not configured():
                raise StorageNotConfigured("R2 isn't configured (R2_BUCKET, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, "
                                           "R2_ENDPOINT or R2_ACCOUNT_ID)")
            import boto3
            from botocore.config import Config
            endpoint = os.environ.get("R2_ENDPOINT") or \
                f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com"
            client = boto3.client("s3", endpoint_url=endpoint, region_name="auto",
                                  aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
                                  aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
                                  config=Config(signature_version="s3v4", retries={"max_attempts": 5}))
        self.client = client
        self.bucket = bucket or os.environ["R2_BUCKET"]

    # ---- uploads ----

    def presign_put(self, key: str, expires: int = 3600) -> str:
        return self.client.generate_presigned_url("put_object", ExpiresIn=expires, HttpMethod="PUT",
                                                  Params={"Bucket": self.bucket, "Key": key,
                                                          "ContentType": "application/pdf"})

    def head(self, key: str) -> dict | None:
        from botocore.exceptions import ClientError
        try:
            return self.client.head_object(Bucket=self.bucket, Key=key)
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
                return None
            raise

    def read_start(self, key: str, n: int = 1024) -> bytes:
        return self.client.get_object(Bucket=self.bucket, Key=key, Range=f"bytes=0-{n - 1}")["Body"].read()

    def download(self, key: str, path: str) -> str:
        """Stream the object to `path`; returns its sha256 (hex)."""
        digest = hashlib.sha256()
        body = self.client.get_object(Bucket=self.bucket, Key=key)["Body"]
        with open(path, "wb") as f:
            for chunk in iter(lambda: body.read(1024 * 1024), b""):
                digest.update(chunk)
                f.write(chunk)
        return digest.hexdigest()

    # ---- previews ----

    def put_bytes(self, key: str, data: bytes, content_type: str) -> None:
        self.client.put_object(Bucket=self.bucket, Key=key, Body=data, ContentType=content_type)

    def presign_get(self, key: str, expires: int = 900) -> str:
        return self.client.generate_presigned_url("get_object", ExpiresIn=expires,
                                                  Params={"Bucket": self.bucket, "Key": key})

    def delete_issue(self, pdf_key: str) -> int:
        """The original and its previews."""
        keys = [pdf_key]
        prefix = pdf_key.removesuffix(".pdf") + "/"
        for page in self.client.get_paginator("list_objects_v2").paginate(Bucket=self.bucket, Prefix=prefix):
            keys += [o["Key"] for o in page.get("Contents", [])]
        for i in range(0, len(keys), 1000):
            self.client.delete_objects(Bucket=self.bucket,
                                       Delete={"Objects": [{"Key": k} for k in keys[i:i + 1000]], "Quiet": True})
        return len(keys)
