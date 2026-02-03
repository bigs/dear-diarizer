"""
Checkpoint tarball uploads to object storage (S3/GCS).

This module provides a best-effort, async uploader that tars checkpoint
directories and uploads them to a bucket when configured via env vars.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path
import queue
import tarfile
import tempfile
import threading
import time
from typing import Optional
from urllib.parse import urlparse


LOGGER = logging.getLogger(__name__)


def parse_bucket_uri(uri: str) -> tuple[str, str, str]:
    """Parse an s3:// or gs:// bucket URI into (scheme, bucket, prefix)."""
    parsed = urlparse(uri)
    scheme = parsed.scheme
    bucket = parsed.netloc
    if scheme not in {"s3", "gs"}:
        raise ValueError(f"Unsupported bucket scheme: {scheme!r}")
    if not bucket:
        raise ValueError("Bucket URI must include a bucket name.")
    prefix = parsed.path.lstrip("/")
    if prefix.endswith("/"):
        prefix = prefix.rstrip("/")
    return scheme, bucket, prefix


def _has_aws_credentials() -> bool:
    return bool(os.getenv("AWS_ACCESS_KEY_ID")) and bool(
        os.getenv("AWS_SECRET_ACCESS_KEY")
    )


def _has_gcp_credentials() -> bool:
    return bool(os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))


def _dir_snapshot(path: Path) -> tuple[int, int]:
    file_count = 0
    total_size = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            file_count += 1
            try:
                total_size += (Path(root) / name).stat().st_size
            except OSError:
                # File disappeared mid-walk; ignore for snapshot stability check.
                continue
    return file_count, total_size


def build_checkpoint_tarball(
    checkpoint_dir: Path,
    checkpoint_subdir: Path,
    tarball_path: Path,
) -> None:
    """Create a tar.gz containing configs and the checkpoint subdir."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_subdir = Path(checkpoint_subdir)
    tarball_path = Path(tarball_path)

    training_config = checkpoint_dir / "training_config.json"
    model_config = checkpoint_dir / "model_config.json"

    with tarfile.open(tarball_path, "w:gz") as tar:
        if training_config.exists():
            tar.add(training_config, arcname=training_config.name)
        if model_config.exists():
            tar.add(model_config, arcname=model_config.name)
        if checkpoint_subdir.exists():
            rel_subdir = checkpoint_subdir.relative_to(checkpoint_dir)
            tar.add(checkpoint_subdir, arcname=str(rel_subdir))


class BaseUploader:
    """Uploader interface for bucket providers."""

    def upload_file(self, local_path: Path, key: str) -> None:
        raise NotImplementedError

    def upload_text(self, key: str, text: str) -> None:
        raise NotImplementedError


class S3Uploader(BaseUploader):
    def __init__(self, bucket: str) -> None:
        import boto3

        self._bucket = bucket
        self._client = boto3.client("s3")

    def upload_file(self, local_path: Path, key: str) -> None:
        self._client.upload_file(str(local_path), self._bucket, key)

    def upload_text(self, key: str, text: str) -> None:
        self._client.put_object(
            Bucket=self._bucket,
            Key=key,
            Body=text.encode("utf-8"),
            ContentType="text/plain",
        )


class GCSUploader(BaseUploader):
    def __init__(self, bucket: str) -> None:
        from google.cloud import storage

        self._client = storage.Client()
        self._bucket = self._client.bucket(bucket)

    def upload_file(self, local_path: Path, key: str) -> None:
        blob = self._bucket.blob(key)
        blob.upload_from_filename(str(local_path))

    def upload_text(self, key: str, text: str) -> None:
        blob = self._bucket.blob(key)
        blob.upload_from_string(text, content_type="text/plain")


def _create_uploader(scheme: str, bucket: str) -> BaseUploader:
    if scheme == "s3":
        return S3Uploader(bucket)
    if scheme == "gs":
        return GCSUploader(bucket)
    raise ValueError(f"Unsupported bucket scheme: {scheme!r}")


@dataclass(frozen=True)
class CheckpointUploadTask:
    step: int
    checkpoint_subdir: Path
    tarball_name: str
    update_best_pointer: bool


class CheckpointUploadManager:
    """Async tarball+upload manager for checkpoint directories."""

    def __init__(
        self,
        checkpoint_dir: Path,
        scheme: str,
        bucket: str,
        prefix: str,
        uploader: Optional[BaseUploader] = None,
        logger: Optional[logging.Logger] = None,
        start_worker: bool = True,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.scheme = scheme
        self.bucket = bucket
        self.prefix = prefix
        self._uploader = uploader or _create_uploader(scheme, bucket)
        self._logger = logger or LOGGER
        self._queue: queue.Queue[CheckpointUploadTask | None] = queue.Queue()
        self._closed = False
        self._thread: Optional[threading.Thread] = None

        if start_worker:
            self._thread = threading.Thread(
                target=self._worker,
                name="checkpoint-upload-worker",
                daemon=True,
            )
            self._thread.start()

    @classmethod
    def from_env(
        cls,
        checkpoint_dir: Path,
        logger: Optional[logging.Logger] = None,
    ) -> Optional["CheckpointUploadManager"]:
        bucket_uri = os.getenv("CHECKPOINT_BUCKET_PATH")
        if not bucket_uri:
            return None

        log = logger or LOGGER
        try:
            scheme, bucket, prefix = parse_bucket_uri(bucket_uri)
        except ValueError as exc:
            log.warning("Invalid CHECKPOINT_BUCKET_PATH %r: %s", bucket_uri, exc)
            return None

        if scheme == "s3" and not _has_aws_credentials():
            log.warning(
                "CHECKPOINT_BUCKET_PATH is s3:// but AWS credentials are missing."
            )
            return None
        if scheme == "gs" and not _has_gcp_credentials():
            log.warning(
                "CHECKPOINT_BUCKET_PATH is gs:// but GOOGLE_APPLICATION_CREDENTIALS is missing."
            )
            return None

        try:
            return cls(
                checkpoint_dir=checkpoint_dir,
                scheme=scheme,
                bucket=bucket,
                prefix=prefix,
                logger=log,
            )
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("Failed to initialize checkpoint uploader: %s", exc)
            return None

    def _object_key(self, name: str) -> str:
        if self.prefix:
            return f"{self.prefix}/{name}"
        return name

    def _best_pointer_key(self) -> str:
        return self._object_key("best")

    def _tarball_name(self, step: int, is_best: bool) -> str:
        base = self.checkpoint_dir.name
        kind = "best" if is_best else "checkpoint"
        return f"{base}-{kind}-step-{step:08d}.tar.gz"

    def _make_task(self, step: int, is_best: bool) -> CheckpointUploadTask:
        subdir = "best" if is_best else "checkpoints"
        checkpoint_subdir = self.checkpoint_dir / subdir / str(step)
        tarball_name = self._tarball_name(step, is_best)
        return CheckpointUploadTask(
            step=step,
            checkpoint_subdir=checkpoint_subdir,
            tarball_name=tarball_name,
            update_best_pointer=is_best,
        )

    def enqueue_checkpoint(self, step: int, *, is_best: bool) -> None:
        if self._closed:
            return
        task = self._make_task(step, is_best)
        if self._thread is None:
            self._handle_task(task)
        else:
            self._queue.put(task)

    def wait_until_finished(self) -> None:
        if self._closed:
            return
        if self._thread is None:
            return
        self._queue.join()

    def shutdown(self) -> None:
        if self._closed:
            return
        if self._thread is None:
            self._closed = True
            return
        self._queue.join()
        self._queue.put(None)
        self._thread.join()
        self._closed = True

    def _wait_for_checkpoint_dir(
        self,
        path: Path,
        timeout_s: float = 300.0,
        stable_checks: int = 2,
        interval_s: float = 1.0,
    ) -> None:
        start = time.time()
        last_snapshot: Optional[tuple[int, int]] = None
        stable_count = 0

        while True:
            if path.exists():
                snapshot = _dir_snapshot(path)
                if snapshot == last_snapshot:
                    stable_count += 1
                    if stable_count >= stable_checks:
                        return
                else:
                    stable_count = 0
                    last_snapshot = snapshot

            if time.time() - start > timeout_s:
                self._logger.warning(
                    "Timed out waiting for checkpoint %s to stabilize; proceeding.",
                    path,
                )
                return

            time.sleep(interval_s)

    def _handle_task(self, task: CheckpointUploadTask) -> None:
        try:
            self._wait_for_checkpoint_dir(task.checkpoint_subdir)
            with tempfile.TemporaryDirectory() as tmpdir:
                tarball_path = Path(tmpdir) / task.tarball_name
                build_checkpoint_tarball(
                    checkpoint_dir=self.checkpoint_dir,
                    checkpoint_subdir=task.checkpoint_subdir,
                    tarball_path=tarball_path,
                )
                object_key = self._object_key(task.tarball_name)
                self._uploader.upload_file(tarball_path, object_key)
                if task.update_best_pointer:
                    self._uploader.upload_text(
                        self._best_pointer_key(),
                        f"{task.tarball_name}\n",
                    )
        except Exception as exc:
            self._logger.warning("Checkpoint upload failed: %s", exc)

    def _worker(self) -> None:
        while True:
            task = self._queue.get()
            try:
                if task is None:
                    return
                self._handle_task(task)
            finally:
                self._queue.task_done()
