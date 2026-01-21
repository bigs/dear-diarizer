# VoxCeleb Evaluation

This folder contains evaluation utilities for WavLeJEPA on VoxCeleb.

## Requirements

- VoxCeleb1 audio tree (e.g., `voxceleb1/wav/idxxxx/.../*.wav`)
- VoxCeleb1 verification trial list (e.g., `veri_test.txt`)

## Speaker Identification (Linear Probe)

```bash
python -m evals.voxceleb.probe \
  --checkpoint checkpoints-batch128-conservative \
  --voxceleb-root /path/to/voxceleb1/wav \
  --batch-size 32 \
  --max-duration 10.0 \
  --output results/voxceleb_probe.json
```

## Speaker Verification (EER Calibration)

```bash
python -m evals.voxceleb.verify \
  --checkpoint checkpoints-batch128-conservative \
  --voxceleb-root /path/to/voxceleb1/wav \
  --trials /path/to/veri_test.txt \
  --batch-size 32 \
  --max-duration 10.0 \
  --debug-stats \
  --output results/voxceleb_verif.json
```

Notes:
- `--trials` should point to the official VoxCeleb1 verification list.
- Embeddings are cached under `results/voxceleb_verif_cache` by default; pass
  `--force-recompute` to ignore the cache.
