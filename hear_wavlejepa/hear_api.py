"""
HEAR API implementation for WavLeJEPA.

Provides the three required HEAR functions:
- load_model(model_file_path) -> Model
- get_timestamp_embeddings(audio, model) -> (embeddings, timestamps)
- get_scene_embeddings(audio, model) -> embeddings
"""

from __future__ import annotations

import torch

from .model_torch import WavLeJEPATorch
from .orbax_loader import load_torch_state_dict_from_orbax


def load_model(model_file_path: str) -> WavLeJEPATorch:
    """Load a WavLeJEPA model from an Orbax checkpoint.

    Args:
        model_file_path: Path to checkpoint root or subdirectory

    Returns:
        WavLeJEPATorch model with HEAR attributes:
        - sample_rate: 16000
        - timestamp_embedding_size: 768
        - scene_embedding_size: 768
    """
    # Load state dict and config from Orbax checkpoint
    state_dict, config = load_torch_state_dict_from_orbax(model_file_path)

    # Create PyTorch model
    model = WavLeJEPATorch(config)

    # Load weights
    # Use strict=False to allow missing predictor weights (not needed for inference)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    # Validate that we loaded the important weights
    # Missing keys should only be predictor-related or buffers
    critical_missing = [
        k for k in missing
        if not k.startswith("predictor.")
        and "pos_encoding.pe" not in k  # PE is computed, not loaded
    ]
    if critical_missing:
        raise RuntimeError(
            f"Critical weights missing from checkpoint: {critical_missing}"
        )

    model.eval()
    return model


def get_timestamp_embeddings(
    audio: torch.Tensor,
    model: WavLeJEPATorch,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get timestamp (frame-level) embeddings from audio.

    Args:
        audio: Mono audio in [-1, 1], shape (n_sounds, n_samples)

    Returns:
        embeddings: Tensor of shape (n_sounds, n_timestamps, timestamp_embedding_size)
        timestamps: Tensor of shape (n_sounds, n_timestamps) in milliseconds (centered)
    """
    with torch.no_grad():
        # Handle single audio
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)

        batch_size = audio.shape[0]

        # Extract features
        embeddings = model.extract_features(audio)  # (B, T, D)

        # Compute timestamps (centered, in milliseconds)
        # Frame i corresponds to samples [i * stride, i * stride + receptive_field)
        # Center is at i * stride + receptive_field / 2
        stride = model.total_stride
        rf = model.receptive_field
        sample_rate = model.sample_rate

        n_frames = embeddings.shape[1]
        frame_indices = torch.arange(n_frames, device=audio.device, dtype=torch.float32)

        # Center sample for each frame
        center_samples = frame_indices * stride + rf / 2

        # Convert to milliseconds
        timestamps_ms = center_samples / sample_rate * 1000.0

        # Broadcast to batch
        timestamps = timestamps_ms.unsqueeze(0).expand(batch_size, -1)

        return embeddings, timestamps


def get_scene_embeddings(
    audio: torch.Tensor,
    model: WavLeJEPATorch,
) -> torch.Tensor:
    """Get scene (clip-level) embeddings from audio.

    Uses mean pooling over timestamp embeddings.

    Args:
        audio: Mono audio in [-1, 1], shape (n_sounds, n_samples)

    Returns:
        embeddings: Tensor of shape (n_sounds, scene_embedding_size)
    """
    with torch.no_grad():
        # Get timestamp embeddings
        timestamp_embeddings, _ = get_timestamp_embeddings(audio, model)

        # Mean pool over time dimension
        scene_embeddings = timestamp_embeddings.mean(dim=1)

        return scene_embeddings
