# Training Server Checklist (WavLeJEPA)

This is the minimum a freshly launched server needs to run `python -m wavlejepa.train`.

## 0) Pick your accelerator

- **NVIDIA GPU (CUDA 12)**: install NVIDIA driver + CUDA 12 userspace, then `uv sync --group cuda`.
- **TPU**: install libtpu, then `uv sync --group tpu`.
- **CPU-only**: works for smoke tests only (slow), `uv sync`.

## 1) System packages

Required for audio decoding (librosa) and common tooling:

- `ffmpeg`
- `libsndfile` (often packaged as `libsndfile1` + `libsndfile1-dev`)
- `git`, `curl` (for setup)

If you only use pre-decoded `.npy` shards, `ffmpeg/libsndfile` are less critical but still recommended.

## 2) Python + uv

- Python **3.12+** is required (see `pyproject.toml`).
- Install uv and sync deps:

```bash
uv sync                    # CPU/basic
uv sync --group cuda       # NVIDIA CUDA 12
# OR
uv sync --group tpu --find-links https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

## 3) JAX accelerator verification

Sanity checks:

```bash
python - <<'PY'
import jax
print('devices:', jax.devices())
PY
```

Expected: your GPU(s) or TPU(s) appear; CPU-only means the accelerator stack is not installed.

## 4) WandB auth

If you want logging enabled (default):

```bash
wandb login
```

To run without WandB:

```bash
uv run python -m wavlejepa.train --dummy --no-wandb
```

## 5) Data access

Training expects **WebDataset shards** and uses `webdataset` streaming.

- Local shards example:
  - `shards/train-{000000..000099}.tar`
- Remote shards example:
  - `s3://bucket/train-{000000..000999}.tar`
  - `gs://bucket/train-{000000..000999}.tar`

### If using S3 or GCS

You must also provide object-store credentials and filesystem support.

- **S3**: set AWS credentials in env or via `aws configure`.
- **GCS**: `gcloud auth application-default login` or service account JSON.
- **Filesystem adapters**: install `s3fs` for S3 or `gcsfs` for GCS if your environment does not already provide them.

Note: `s3fs/gcsfs` are not declared in `pyproject.toml`, so add them explicitly if you stream from cloud storage.

## 6) Checkpoint directory

Default checkpoint path is `./checkpoints` (configurable in YAML). Ensure:

- The directory exists or is creatable.
- Enough disk space for Orbax checkpoints.

## 7) Environment knobs (optional but common)

- Control JAX memory preallocation:

```bash
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4
```

## 8) Example run

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 uv run python -m wavlejepa.train \
  --config configs/spark-batch128-conservative.yaml \
  --shards "shards/train-{000000..000099}.tar"
```

## 9) Preprocessing (optional)

If you want to pre-decode audio into tensor shards (faster training):

```bash
uv run python -m wavlejepa.prepare_shards \
  --input downloads/ \
  --output shards/train-%06d.tar
```

---

## Quick Failure Triage

- **Only CPU shows in `jax.devices()`**: accelerator stack not installed or not visible to the process.
- **Audio decode errors**: missing `ffmpeg`/`libsndfile` or unsupported audio format.
- **`s3://`/`gs://` errors**: missing credentials and/or `s3fs`/`gcsfs`.
- **WandB errors**: missing `wandb login`, or set `--no-wandb` to proceed.
