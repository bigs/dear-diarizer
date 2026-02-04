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
    checkpoint_dir: Path | str,
    checkpoint_subdir: Path | str,
    tarball_path: Path | str,
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


def checkpoint_tarball_name(
    checkpoint_dir: Path | str, *, step: int, is_best: bool
) -> str:
    """Build a checkpoint tarball name matching the training uploader format."""
    checkpoint_dir = Path(checkpoint_dir)
    kind = "best" if is_best else "checkpoint"
    return f"{checkpoint_dir.name}-{kind}-step-{step:08d}.tar.gz"


def _object_key(prefix: str, name: str) -> str:
    if prefix:
        return f"{prefix}/{name}"
    return name


def resolve_checkpoint_ref(
    checkpoint_path: Path | str,
    *,
    step: int | None = None,
    is_best: bool | None = None,
    latest: bool = False,
) -> tuple[Path, int, bool]:
    """
    Resolve a user-provided checkpoint path to (checkpoint_dir, step, is_best).

    Accepted paths:
    - /path/to/run/checkpoints/<step>
    - /path/to/run/best/<step>
    - /path/to/run/checkpoints or /path/to/run/best (requires --step or --latest)
    - /path/to/run (requires --step or --latest)
    """
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {path}")
    if path.is_file():
        raise ValueError(f"Checkpoint path must be a directory, got file: {path}")

    if path.name.isdigit() and path.parent.name in {"checkpoints", "best"}:
        inferred_step = int(path.name)
        inferred_best = path.parent.name == "best"
        if step is not None and step != inferred_step:
            raise ValueError(
                f"Conflicting step: inferred {inferred_step} from {path}, got {step}."
            )
        if is_best is not None and is_best != inferred_best:
            raise ValueError(
                f"Conflicting --best: inferred {inferred_best} from {path}."
            )
        return path.parent.parent, inferred_step, inferred_best

    def choose_step(subdir: Path) -> int:
        if step is not None:
            step_dir = subdir / str(step)
            if not step_dir.is_dir():
                raise FileNotFoundError(
                    f"Checkpoint step directory not found: {step_dir}"
                )
            return step
        if not latest:
            raise ValueError(
                f"Checkpoint step not specified for {path}. Provide --step or --latest."
            )
        candidates = [
            int(p.name) for p in subdir.iterdir() if p.is_dir() and p.name.isdigit()
        ]
        if not candidates:
            raise FileNotFoundError(f"No checkpoint steps found under {subdir}")
        return max(candidates)

    if path.name in {"checkpoints", "best"}:
        checkpoint_dir = path.parent
        inferred_best = path.name == "best"
        if is_best is not None and is_best != inferred_best:
            raise ValueError(
                f"Conflicting --best: inferred {inferred_best} from {path}."
            )
        chosen_step = choose_step(path)
        return checkpoint_dir, chosen_step, inferred_best

    checkpoint_dir = path
    chosen_best = bool(is_best) if is_best is not None else False
    subdir = checkpoint_dir / ("best" if chosen_best else "checkpoints")
    if not subdir.exists():
        if (
            is_best is None
            and (checkpoint_dir / "best").exists()
            and not (checkpoint_dir / "checkpoints").exists()
        ):
            chosen_best = True
            subdir = checkpoint_dir / "best"
        else:
            raise FileNotFoundError(f"Missing checkpoint subdir: {subdir}")

    chosen_step = choose_step(subdir)
    return checkpoint_dir, chosen_step, chosen_best


def upload_checkpoint_tarball(
    checkpoint_path: Path | str,
    *,
    bucket_uri: str,
    step: int | None = None,
    is_best: bool | None = None,
    latest: bool = False,
    wait_for_stable: bool = True,
    timeout_s: float = 300.0,
    uploader: Optional["BaseUploader"] = None,
    logger: Optional[logging.Logger] = None,
) -> str:
    """
    Build and upload a checkpoint tarball matching the training uploader format.

    Returns the tarball name that was uploaded.
    """
    log = logger or LOGGER
    checkpoint_dir, resolved_step, resolved_best = resolve_checkpoint_ref(
        checkpoint_path, step=step, is_best=is_best, latest=latest
    )
    checkpoint_subdir = (
        checkpoint_dir
        / ("best" if resolved_best else "checkpoints")
        / str(resolved_step)
    )
    tarball_name = checkpoint_tarball_name(
        checkpoint_dir, step=resolved_step, is_best=resolved_best
    )

    training_config = checkpoint_dir / "training_config.json"
    model_config = checkpoint_dir / "model_config.json"
    if not training_config.exists() or not model_config.exists():
        log.warning(
            "Missing config JSONs in %s (training_config.json=%s, model_config.json=%s).",
            checkpoint_dir,
            training_config.exists(),
            model_config.exists(),
        )

    scheme, bucket, prefix = parse_bucket_uri(bucket_uri)
    upload_impl = uploader or _create_uploader(scheme, bucket)

    if wait_for_stable:
        manager = CheckpointUploadManager(
            checkpoint_dir=checkpoint_dir,
            scheme=scheme,
            bucket=bucket,
            prefix=prefix,
            uploader=upload_impl,
            logger=log,
            start_worker=False,
        )
        manager._wait_for_checkpoint_dir(checkpoint_subdir, timeout_s=timeout_s)

    with tempfile.TemporaryDirectory() as tmpdir:
        tarball_path = Path(tmpdir) / tarball_name
        build_checkpoint_tarball(
            checkpoint_dir=checkpoint_dir,
            checkpoint_subdir=checkpoint_subdir,
            tarball_path=tarball_path,
        )
        object_key = _object_key(prefix, tarball_name)
        upload_impl.upload_file(tarball_path, object_key)
        if resolved_best:
            upload_impl.upload_text(_object_key(prefix, "best"), f"{tarball_name}\n")

    return tarball_name


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
        checkpoint_dir: Path | str,
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
        checkpoint_dir: Path | str,
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
        return _object_key(self.prefix, name)

    def _best_pointer_key(self) -> str:
        return self._object_key("best")

    def _tarball_name(self, step: int, is_best: bool) -> str:
        return checkpoint_tarball_name(self.checkpoint_dir, step=step, is_best=is_best)

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

    def shutdown(self) -> None:
        if self._closed:
            return
        if self._thread is None:
            self._closed = True
            return
        self._closed = True
        self._queue.join()
        self._queue.put(None)
        self._thread.join()

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
            else:
                stable_count = 0
                last_snapshot = None

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
