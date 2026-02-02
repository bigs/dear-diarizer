from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import soundfile as sf

import jax


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evals.voxceleb.checkpoint_utils import resolve_checkpoint_root, restore_model
from evals.voxceleb.embeddings import extract_embeddings
from wavlejepa.model import WavLeJEPAConfig
from wavlejepa.training.checkpoint import WavLeJEPACheckpointer
from wavlejepa.training.config import CheckpointConfig, OptimizerConfig, TrainingConfig
from wavlejepa.training.state import create_train_state


def _write_wav(path: Path, duration_s: float = 0.5, sr: int = 16000) -> None:
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    audio = 0.1 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
    sf.write(path, audio, sr)


def _make_checkpoint(tmp_path: Path, *, save_best: bool) -> Path:
    checkpoint_dir = tmp_path / ("ckpt_best" if save_best else "ckpt_latest")
    checkpoint_config = CheckpointConfig(
        checkpoint_dir=str(checkpoint_dir),
        save_every_n_steps=1,
        keep_n_checkpoints=1,
        save_best=save_best,
    )
    optimizer_config = OptimizerConfig(
        peak_lr=1e-4,
        warmup_steps=1,
        total_steps=2,
        weight_decay=0.0,
    )
    training_config = TrainingConfig(
        optimizer=optimizer_config,
        checkpoint=checkpoint_config,
    )
    model_config = WavLeJEPAConfig(
        waveform_embed_dim=16,
        waveform_num_groups=1,
        context_embed_dim=16,
        context_num_heads=4,
        context_num_layers=2,
        context_ffn_dim=32,
        context_dropout=0.0,
        context_top_k_layers=1,
        context_top_k_norm="instance",
        predictor_dim=8,
        predictor_num_heads=2,
        predictor_num_layers=1,
        predictor_ffn_dim=16,
        predictor_dropout=0.0,
        max_seq_len=200,
    )

    state, _ = create_train_state(model_config, optimizer_config, jax.random.key(0))
    checkpointer = WavLeJEPACheckpointer(
        config=checkpoint_config,
        training_config=training_config,
        model_config=model_config,
    )
    checkpointer.save(state)
    if save_best:
        checkpointer.save_best(state, val_loss=0.1)
    checkpointer.wait_until_finished()
    return checkpoint_dir


def test_resolve_checkpoint_root_handles_subdirs(tmp_path: Path) -> None:
    checkpoint_dir = _make_checkpoint(tmp_path, save_best=True)

    assert resolve_checkpoint_root(checkpoint_dir) == checkpoint_dir
    assert resolve_checkpoint_root(checkpoint_dir / "best") == checkpoint_dir
    assert resolve_checkpoint_root(checkpoint_dir / "checkpoints") == checkpoint_dir

    best_step_dirs = list((checkpoint_dir / "best").iterdir())
    assert best_step_dirs, "expected a saved best checkpoint"
    assert resolve_checkpoint_root(best_step_dirs[0]) == checkpoint_dir


def test_restore_model_falls_back_to_latest(tmp_path: Path) -> None:
    checkpoint_dir = _make_checkpoint(tmp_path, save_best=False)
    model = restore_model(checkpoint_dir, key=jax.random.key(123))
    assert model is not None


def test_extract_embeddings_shapes(tmp_path: Path) -> None:
    checkpoint_dir = _make_checkpoint(tmp_path, save_best=True)
    audio_path = tmp_path / "sample.wav"
    _write_wav(audio_path, duration_s=0.5)

    emb_mean = extract_embeddings(
        checkpoint_path=checkpoint_dir / "best",
        audio_paths=[audio_path],
        batch_size=1,
        sample_rate=16000,
        max_duration=0.5,
        pooling="mean",
        feature_source="topk",
    )
    assert emb_mean.shape == (1, 16)
    assert np.isfinite(emb_mean).all()

    emb_meanstd = extract_embeddings(
        checkpoint_path=checkpoint_dir / "best",
        audio_paths=[audio_path],
        batch_size=1,
        sample_rate=16000,
        max_duration=0.5,
        pooling="meanstd",
        feature_source="context",
    )
    assert emb_meanstd.shape == (1, 32)
    assert np.isfinite(emb_meanstd).all()
