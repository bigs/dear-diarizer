"""
Orbax checkpoint loading for WavLeJEPA.

Restores JAX/Equinox model params from Orbax checkpoints
and converts them to PyTorch state_dict format.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

from .mapping import jax_params_to_torch_state_dict
from .model_torch import WavLeJEPAConfigTorch


def resolve_checkpoint_root(checkpoint_path: Path, max_depth: int = 3) -> Path:
    """Resolve the checkpoint root containing training/model config files.

    Mirrors evals/voxceleb/checkpoint_utils.resolve_checkpoint_root.
    """
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


def load_model_config(checkpoint_root: Path) -> WavLeJEPAConfigTorch:
    """Load model config from checkpoint directory."""
    model_config_path = checkpoint_root / "model_config.json"
    with model_config_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    return WavLeJEPAConfigTorch(
        waveform_embed_dim=data.get("waveform_embed_dim", 768),
        waveform_num_groups=data.get("waveform_num_groups", 32),
        context_embed_dim=data.get("context_embed_dim", 768),
        context_num_heads=data.get("context_num_heads", 12),
        context_num_layers=data.get("context_num_layers", 12),
        context_ffn_dim=data.get("context_ffn_dim", 3072),
        context_dropout=data.get("context_dropout", 0.0),
        context_top_k_layers=data.get("context_top_k_layers", 8),
        context_top_k_norm=data.get("context_top_k_norm", "instance"),
        max_seq_len=data.get("max_seq_len", 1000),
    )


def restore_orbax_model_params(checkpoint_root: Path) -> dict:
    """Restore model_params from Orbax checkpoint.

    Uses the existing checkpoint utilities to restore the model,
    then extracts the model params.

    Args:
        checkpoint_root: Path to checkpoint root directory

    Returns:
        The model_params pytree (dict of arrays)
    """
    import equinox as eqx

    # Use the existing restore helper which handles all the complexity
    from evals.voxceleb.checkpoint_utils import restore_model

    model = restore_model(checkpoint_root)
    params = eqx.filter(model, eqx.is_array)

    return params


def load_torch_state_dict_from_orbax(
    checkpoint_path: str | Path,
) -> tuple[dict[str, "torch.Tensor"], WavLeJEPAConfigTorch]:
    """Load PyTorch state_dict from Orbax checkpoint.

    Args:
        checkpoint_path: Path to checkpoint root or subdirectory

    Returns:
        Tuple of (state_dict, config) where state_dict maps to torch.Tensor
    """
    import torch

    checkpoint_root = resolve_checkpoint_root(Path(checkpoint_path))

    # Load model config
    config = load_model_config(checkpoint_root)

    # Restore JAX params
    jax_params = restore_orbax_model_params(checkpoint_root)

    # Convert to PyTorch state_dict format
    numpy_state = jax_params_to_torch_state_dict(jax_params)

    # Convert numpy arrays to torch tensors
    torch_state = {k: torch.from_numpy(v.copy()) for k, v in numpy_state.items()}

    return torch_state, config
