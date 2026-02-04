"""Tests for checkpoint upload utilities."""

import io
from pathlib import Path
import tarfile

import pytest

from wavlejepa.training.checkpoint_upload import (
    BaseUploader,
    CheckpointUploadManager,
    build_checkpoint_tarball,
    checkpoint_tarball_name,
    parse_bucket_uri,
    resolve_checkpoint_ref,
    upload_checkpoint_tarball,
)


class FakeUploader(BaseUploader):
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.texts: dict[str, str] = {}

    def upload_file(self, local_path: Path, key: str) -> None:
        self.files[key] = Path(local_path).read_bytes()

    def upload_text(self, key: str, text: str) -> None:
        self.texts[key] = text


def test_parse_bucket_uri_valid() -> None:
    assert parse_bucket_uri("s3://bucket/path") == ("s3", "bucket", "path")
    assert parse_bucket_uri("gs://bucket") == ("gs", "bucket", "")


@pytest.mark.parametrize(
    "uri",
    [
        "bucket/path",
        "s3:///path",
        "ftp://bucket/path",
    ],
)
def test_parse_bucket_uri_invalid(uri: str) -> None:
    with pytest.raises(ValueError):
        parse_bucket_uri(uri)


def test_tarball_contents(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    step_dir = checkpoint_dir / "checkpoints" / "5"
    step_dir.mkdir(parents=True)

    (checkpoint_dir / "training_config.json").write_text("{}")
    (checkpoint_dir / "model_config.json").write_text("{}")
    (step_dir / "dummy.bin").write_text("data")

    tarball_path = tmp_path / "checkpoint.tar.gz"
    build_checkpoint_tarball(checkpoint_dir, step_dir, tarball_path)

    with tarfile.open(tarball_path, "r:gz") as tar:
        names = set(tar.getnames())

    assert "training_config.json" in names
    assert "model_config.json" in names
    assert "checkpoints/5/dummy.bin" in names


def test_best_pointer_payload(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "ckpt-root"
    step_dir = checkpoint_dir / "best" / "7"
    step_dir.mkdir(parents=True)

    (checkpoint_dir / "training_config.json").write_text("{}")
    (checkpoint_dir / "model_config.json").write_text("{}")
    (step_dir / "dummy.bin").write_text("data")

    uploader = FakeUploader()
    manager = CheckpointUploadManager(
        checkpoint_dir=checkpoint_dir,
        scheme="s3",
        bucket="bucket",
        prefix="runs/1",
        uploader=uploader,
        start_worker=False,
    )

    task = manager._make_task(7, is_best=True)
    manager._handle_task(task)

    assert uploader.texts["runs/1/best"] == f"{task.tarball_name}\n"


def test_resolve_checkpoint_ref_step_dir(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    step_dir = run_dir / "checkpoints" / "5"
    step_dir.mkdir(parents=True)

    checkpoint_dir, step, is_best = resolve_checkpoint_ref(step_dir)
    assert checkpoint_dir == run_dir
    assert step == 5
    assert is_best is False


def test_resolve_checkpoint_ref_latest(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "checkpoints" / "5").mkdir(parents=True)
    (run_dir / "checkpoints" / "12").mkdir(parents=True)

    checkpoint_dir, step, is_best = resolve_checkpoint_ref(
        run_dir / "checkpoints", latest=True
    )
    assert checkpoint_dir == run_dir
    assert step == 12
    assert is_best is False


def test_upload_checkpoint_tarball_checkpoint(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "run"
    step_dir = checkpoint_dir / "checkpoints" / "5"
    step_dir.mkdir(parents=True)
    (checkpoint_dir / "training_config.json").write_text("{}")
    (checkpoint_dir / "model_config.json").write_text("{}")
    (step_dir / "dummy.bin").write_text("data")

    uploader = FakeUploader()
    tarball_name = upload_checkpoint_tarball(
        step_dir,
        bucket_uri="s3://bucket/runs/1",
        uploader=uploader,
        wait_for_stable=False,
    )

    expected_name = checkpoint_tarball_name(checkpoint_dir, step=5, is_best=False)
    assert tarball_name == expected_name

    key = f"runs/1/{tarball_name}"
    assert key in uploader.files
    assert "runs/1/best" not in uploader.texts

    with tarfile.open(fileobj=io.BytesIO(uploader.files[key]), mode="r:gz") as tar:
        names = set(tar.getnames())
    assert "training_config.json" in names
    assert "model_config.json" in names
    assert "checkpoints/5/dummy.bin" in names


def test_upload_checkpoint_tarball_missing_step_raises(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "run"
    (checkpoint_dir / "checkpoints").mkdir(parents=True)
    (checkpoint_dir / "training_config.json").write_text("{}")
    (checkpoint_dir / "model_config.json").write_text("{}")

    uploader = FakeUploader()
    with pytest.raises(FileNotFoundError, match="Checkpoint step directory not found"):
        upload_checkpoint_tarball(
            checkpoint_dir,
            bucket_uri="s3://bucket/runs/1",
            step=999,
            uploader=uploader,
            wait_for_stable=False,
        )


def test_upload_checkpoint_tarball_best_pointer(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "run"
    step_dir = checkpoint_dir / "best" / "7"
    step_dir.mkdir(parents=True)
    (checkpoint_dir / "training_config.json").write_text("{}")
    (checkpoint_dir / "model_config.json").write_text("{}")
    (step_dir / "dummy.bin").write_text("data")

    uploader = FakeUploader()
    tarball_name = upload_checkpoint_tarball(
        step_dir,
        bucket_uri="s3://bucket/runs/1",
        uploader=uploader,
        wait_for_stable=False,
    )

    expected_name = checkpoint_tarball_name(checkpoint_dir, step=7, is_best=True)
    assert tarball_name == expected_name
    assert uploader.texts["runs/1/best"] == f"{tarball_name}\n"


def test_env_gating(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("CHECKPOINT_BUCKET_PATH", raising=False)
    assert CheckpointUploadManager.from_env(tmp_path) is None

    monkeypatch.setenv("CHECKPOINT_BUCKET_PATH", "s3://bucket/path")
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    assert CheckpointUploadManager.from_env(tmp_path) is None
