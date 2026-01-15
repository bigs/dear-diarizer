# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

WavLeJEPA is a JAX/Equinox implementation of a self-supervised audio representation learning model combining:
- **WavJEPA**: Time-domain waveform processing (raw audio → embeddings at 100Hz)
- **LeJEPA**: SIGReg regularization for heuristics-free training (no EMA teacher, stop-gradient, etc.)

## Commands

### Setup
```bash
uv sync                    # CPU/basic install
uv sync --group dev        # Include dev tools (ruff, pyright)
uv sync --group cuda       # CUDA 12 support
uv sync --group tpu --find-links https://storage.googleapis.com/jax-releases/libtpu_releases.html  # TPU
```

### Training
```bash
# Prepare data shards (preprocesses audio to tensors)
uv run python -m wavlejepa.prepare_shards --input downloads/ --output shards/train-%06d.tar

# Training with config
XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 uv run python -m wavlejepa.train \
    --config configs/spark-batch128-conservative.yaml \
    --shards "shards/train-{000000..000099}.tar"

# Quick test with dummy data
uv run python -m wavlejepa.train --dummy --no-wandb
```

### Linting
```bash
uv run ruff check .
uv run pyright
```

## Architecture

### Model Components (`wavlejepa/`)
```
WavLeJEPA (model.py)
├── WaveformEncoder    - Conv stack: raw audio [T] → embeddings [N, 768] at 100Hz
├── ContextEncoder     - Transformer: embeddings → contextualized representations
├── Predictor          - Cross-attention: predicts masked targets from context
└── Projector          - MLP: maps to SIGReg space (training only)
```

### Training Flow
1. **Masking**: Sample context blocks (~10% of frames) and target blocks (non-overlapping)
2. **Encoding**: Process context-masked sequence through ContextEncoder
3. **Prediction**: Predictor uses cross-attention to predict target representations
4. **Loss**: Invariance loss (L2 prediction error) + SIGReg loss (isotropic Gaussian regularizer)

### Loss Functions (`losses.py`, `sigreg.py`)
- **Invariance loss**: MSE between predicted and actual target representations (averaged over batch)
- **SIGReg loss**: Epps-Pulley test statistic measuring deviation from N(0,I). Intentionally scales with batch size (× N) per the statistical test formulation

### Training Infrastructure (`training/`)
- `config.py`: Dataclass configs loadable from YAML
- `state.py`: TrainState (Equinox module), optimizer creation with warmup+cosine decay
- `step.py`: JIT-compiled train/eval steps with multi-GPU data parallelism
- `checkpoint.py`: Orbax async checkpointing with best model tracking
- `logging.py`: Wandb integration

### Data Pipeline (`data.py`, `prepare_shards.py`)
- WebDataset for streaming from S3/GCS/local
- `prepare_shards.py`: Pre-decode audio to tensor shards (eliminates MP3 decode bottleneck)

## Key Training Dynamics

**Learning rate sensitivity**: Peak LR is critical. If too high, model finds good minimum during warmup then gets kicked out at peak. SIGReg spiking (>1000) indicates embedding space destabilization.

**Healthy training signals**:
- SIGReg: 300-600 range, stable
- Invariance: Decreasing, some volatility OK
- Total loss: Steady decline

**Configs in `configs/`** are hardware-specific (batch size, memory preallocation). Conservative configs use lower peak LR (1.5e-4) and longer warmup (7500 steps).

## Graceful Shutdown

Training supports SIGINT (Ctrl+C) graceful shutdown:
- First Ctrl+C: Finishes current step, saves checkpoint, closes wandb
- Second Ctrl+C: Immediate exit

## Issue Tracking

See `AGENTS.md` for issue tracking with **bd (beads)**. Run `bd prime` for full workflow context.
