"""Checkpoint helpers for VoxCeleb evaluation."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Optional

import jax

from wavlejepa.model import WavLeJEPA, WavLeJEPAConfig
from wavlejepa.training.checkpoint import WavLeJEPACheckpointer
from wavlejepa.training.config import TrainingConfig


def resolve_checkpoint_root(checkpoint_path: Path, max_depth: int = 3) -> Path:
    """Resolve the checkpoint root containing training/model config files."""
    path = Path(checkpoint_path)
    if path.is_file():
        path = path.parent

    current = path
    for _ in range(max_depth + 1):
        training_config = current / "training_config.json"
        model_config = current / "model_config.json"
        if training_config.exists() and model_config.exists():
            return current
        if current.parent == current:
            break
        current = current.parent

    raise FileNotFoundError(
        f"Could not find training_config.json and model_config.json starting from {path}"
    )


def _load_configs(checkpoint_root: Path) -> tuple[TrainingConfig, WavLeJEPAConfig]:
    training_config_path = checkpoint_root / "training_config.json"
    model_config_path = checkpoint_root / "model_config.json"

    training_config = TrainingConfig.from_json(training_config_path)
    with model_config_path.open("r", encoding="utf-8") as f:
        model_config = WavLeJEPAConfig.from_dict(json.load(f))

    # Ensure the checkpoint dir matches the resolved root (handles relocated checkpoints).
    checkpoint_config = replace(
        training_config.checkpoint,
        checkpoint_dir=str(checkpoint_root),
    )
    training_config = replace(training_config, checkpoint=checkpoint_config)

    return training_config, model_config


def restore_model(
    checkpoint_path: Path,
    *,
    key: Optional[jax.Array] = None,
) -> WavLeJEPA:
    """Restore a model from a checkpoint path.

    Prefers the best checkpoint if available, otherwise falls back to the latest.
    """
    checkpoint_root = resolve_checkpoint_root(checkpoint_path)
    training_config, model_config = _load_configs(checkpoint_root)

    checkpointer = WavLeJEPACheckpointer(
        config=training_config.checkpoint,
        training_config=training_config,
        model_config=model_config,
    )

    if key is None:
        key = jax.random.key(0)

    result = checkpointer.restore_best(key=key)
    if result is None:
        result = checkpointer.restore(key=key)
    if result is None:
        raise ValueError(f"No checkpoints found under {checkpoint_root}")

    state, _ = result
    return state.model
