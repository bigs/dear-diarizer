# Encoder SIGReg Sweep Instructions

This sweep evaluates whether applying SIGReg directly to encoder embeddings reduces collapse/anisotropy.

## Prereqs
- VoxCeleb1 audio tree: `voxceleb1/wav/`
- VoxCeleb1 verification trials: `eval-downloads/veri_test.txt`
- Training shards (example): `shards/train-{000000..000099}.tar`

## Configs
Sweep configs are located in `configs/encoder-collapse-sweep/`:
- `spark-batch128-conservative-encsig0p0.yaml` (baseline)
- `spark-batch128-conservative-encsig0p05.yaml`
- `spark-batch128-conservative-encsig0p1.yaml`
- `spark-batch128-conservative-encsig0p2.yaml`

Each config writes to a unique `checkpoint_dir`.

## Train (short runs)

Note: Ctrl+C triggers a graceful checkpoint save. For quick feedback, stop around 10k–20k steps.

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 uv run python -m wavlejepa.train \
  --config configs/encoder-collapse-sweep/spark-batch128-conservative-encsig0p0.yaml \
  --shards "shards/train-{000000..000099}.tar"

XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 uv run python -m wavlejepa.train \
  --config configs/encoder-collapse-sweep/spark-batch128-conservative-encsig0p05.yaml \
  --shards "shards/train-{000000..000099}.tar"

XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 uv run python -m wavlejepa.train \
  --config configs/encoder-collapse-sweep/spark-batch128-conservative-encsig0p1.yaml \
  --shards "shards/train-{000000..000099}.tar"

XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 uv run python -m wavlejepa.train \
  --config configs/encoder-collapse-sweep/spark-batch128-conservative-encsig0p2.yaml \
  --shards "shards/train-{000000..000099}.tar"
```

## Compare (verification + collapse)

```bash
uv run python -m evals.voxceleb.compare \
  --checkpoint checkpoints-batch128-conservative-encsig0p0 \
  --voxceleb-root voxceleb1/wav \
  --trials eval-downloads/veri_test.txt \
  --pooling meanstd \
  --feature-source context \
  --output results/voxceleb_compare_encsig0p0.json

uv run python -m evals.voxceleb.compare \
  --checkpoint checkpoints-batch128-conservative-encsig0p05 \
  --voxceleb-root voxceleb1/wav \
  --trials eval-downloads/veri_test.txt \
  --pooling meanstd \
  --feature-source context \
  --output results/voxceleb_compare_encsig0p05.json

uv run python -m evals.voxceleb.compare \
  --checkpoint checkpoints-batch128-conservative-encsig0p1 \
  --voxceleb-root voxceleb1/wav \
  --trials eval-downloads/veri_test.txt \
  --pooling meanstd \
  --feature-source context \
  --output results/voxceleb_compare_encsig0p1.json

uv run python -m evals.voxceleb.compare \
  --checkpoint checkpoints-batch128-conservative-encsig0p2 \
  --voxceleb-root voxceleb1/wav \
  --trials eval-downloads/veri_test.txt \
  --pooling meanstd \
  --feature-source context \
  --output results/voxceleb_compare_encsig0p2.json
```

## Optional quick baselines

Top-K features (no retrain) for a baseline checkpoint:
```bash
uv run python -m evals.voxceleb.compare \
  --checkpoint checkpoints-batch128-conservative \
  --voxceleb-root voxceleb1/wav \
  --trials eval-downloads/veri_test.txt \
  --pooling mean \
  --feature-source topk \
  --output results/voxceleb_compare_baseline_topk.json
```

Collapse-only spot check:
```bash
uv run python -m evals.voxceleb.collapse_check \
  --checkpoint checkpoints-batch128-conservative \
  --voxceleb-root voxceleb1/wav \
  --num-files 100 \
  --frames-per-file 50 \
  --num-pairs 50000 \
  --feature-source topk
```
