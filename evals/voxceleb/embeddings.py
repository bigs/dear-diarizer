"""Extract embeddings from frozen WavLeJEPA checkpoint."""

import json
import os
from pathlib import Path

# Set JAX to allocate GPU memory as needed (not pre-allocate)
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.25")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import equinox as eqx
import librosa
import numpy as np
from tqdm import tqdm

from wavlejepa.model import WavLeJEPA, WavLeJEPAConfig
from wavlejepa.training.checkpoint import WavLeJEPACheckpointer
from wavlejepa.training.config import TrainingConfig


def load_audio_padded(
    path: Path,
    sample_rate: int = 16000,
    max_duration: float = 10.0,
) -> tuple[np.ndarray, int]:
    """Load audio file and pad/crop to fixed length.

    Args:
        path: Audio file path
        sample_rate: Target sample rate
        max_duration: Fixed duration in seconds (pads shorter, crops longer)

    Returns:
        Tuple of (audio waveform with fixed length, original sample length)
    """
    target_length = int(sample_rate * max_duration)
    audio, _ = librosa.load(path, sr=sample_rate, mono=True, duration=max_duration)
    original_length = len(audio)

    # Pad if shorter than target
    if len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
    # Crop if longer (shouldn't happen with duration limit, but safety)
    elif len(audio) > target_length:
        audio = audio[:target_length]

    return audio.astype(np.float32), original_length


def _extract_frame_features(
    model: WavLeJEPA,
    audio: jnp.ndarray,
    feature_source: str,
) -> jnp.ndarray:
    if feature_source == "topk":
        return model.extract_features(audio)  # [frames, 768]
    if feature_source == "context":
        frame_embeddings = model.waveform_encoder(audio)  # [frames, 768]
        return model.context_encoder(frame_embeddings, inference=True)  # [frames, 768]
    raise ValueError(f"Unsupported feature_source: {feature_source}")


def _pool_frames(
    frame_embeddings: jnp.ndarray,
    valid_frames: jnp.ndarray,
    pooling: str,
) -> jnp.ndarray:
    total_frames = frame_embeddings.shape[0]
    valid_frames = jnp.clip(valid_frames, 1, total_frames)
    mask = (jnp.arange(total_frames) < valid_frames).astype(frame_embeddings.dtype)
    masked = frame_embeddings * mask[:, None]
    denom = jnp.maximum(valid_frames, 1).astype(frame_embeddings.dtype)

    mean = jnp.sum(masked, axis=0) / denom
    if pooling == "mean":
        return mean
    if pooling == "meanstd":
        mean_sq = jnp.sum(masked * masked, axis=0) / denom
        var = jnp.maximum(mean_sq - mean * mean, 0.0)
        std = jnp.sqrt(var + 1e-6)
        return jnp.concatenate([mean, std], axis=0)
    raise ValueError(f"Unsupported pooling: {pooling}")


def _extract_single_embedding(
    model: WavLeJEPA,
    audio: jnp.ndarray,
    sample_length: jnp.ndarray,
    pooling: str,
    feature_source: str,
) -> jnp.ndarray:
    """Extract utterance embedding from single audio (for vmap)."""
    frame_embeddings = _extract_frame_features(model, audio, feature_source)
    valid_frames = model.waveform_encoder.output_length(sample_length)
    return _pool_frames(frame_embeddings, valid_frames, pooling)


@eqx.filter_jit
def extract_batch_embeddings(
    model: WavLeJEPA,
    audio_batch: jnp.ndarray,
    lengths: jnp.ndarray,
    pooling: str = "mean",
    feature_source: str = "topk",
) -> jnp.ndarray:
    """Extract embeddings for a batch of audio.

    Args:
        model: Frozen WavLeJEPA model
        audio_batch: Batch of audio waveforms [B, T] (fixed length)
        lengths: Original (unpadded) lengths in samples [B]
        pooling: Pooling strategy ("mean" or "meanstd")
        feature_source: Feature source ("topk" or "context")

    Returns:
        Batch of embeddings [B, 768]
    """
    # vmap over the batch dimension
    batched_extract = jax.vmap(
        lambda audio, length: _extract_single_embedding(
            model, audio, length, pooling, feature_source
        )
    )
    return batched_extract(audio_batch, lengths)


def extract_embeddings(
    checkpoint_path: Path,
    audio_paths: list[Path],
    batch_size: int = 32,
    sample_rate: int = 16000,
    max_duration: float = 10.0,
    pooling: str = "mean",
    feature_source: str = "topk",
) -> np.ndarray:
    """Extract embeddings for a list of audio files.

    Args:
        checkpoint_path: Path to WavLeJEPA checkpoint directory
        audio_paths: List of audio file paths
        batch_size: Batch size (currently processes sequentially)
        sample_rate: Audio sample rate
        max_duration: Maximum audio duration in seconds (crops longer files)
        pooling: Pooling strategy ("mean" or "meanstd")
        feature_source: Feature source ("topk" or "context")

    Returns:
        Embeddings array [num_utterances, 768]
    """
    checkpoint_path = Path(checkpoint_path)

    if pooling not in {"mean", "meanstd"}:
        raise ValueError(f"Unsupported pooling: {pooling}")
    if feature_source not in {"topk", "context"}:
        raise ValueError(f"Unsupported feature_source: {feature_source}")

    # Load configs
    training_config_path = checkpoint_path / "training_config.json"
    model_config_path = checkpoint_path / "model_config.json"

    # Reconstruct configs
    training_config = TrainingConfig.from_json(training_config_path)

    with open(model_config_path) as f:
        model_config_dict = json.load(f)
        model_config = WavLeJEPAConfig.from_dict(model_config_dict)

    # Load checkpoint
    checkpointer = WavLeJEPACheckpointer(
        config=training_config.checkpoint,
        training_config=training_config,
        model_config=model_config,
    )

    # Restore best model
    result = checkpointer.restore_best(key=jax.random.key(0))
    if result is None:
        raise ValueError(f"No checkpoint found at {checkpoint_path}")

    state, _ = result
    model = state.model

    # Load all audio with fixed length (enables batching)
    print(f"Loading {len(audio_paths)} audio files...")
    all_audio = []
    all_lengths = []
    for audio_path in tqdm(audio_paths, desc="Loading audio"):
        audio, length = load_audio_padded(
            audio_path, sample_rate=sample_rate, max_duration=max_duration
        )
        all_audio.append(audio)
        all_lengths.append(length)
    all_audio = np.stack(all_audio)  # [N, T]
    all_lengths = np.asarray(all_lengths, dtype=np.int32)  # [N]
    print(f"Audio loaded: {all_audio.shape}")

    # Extract embeddings in batches
    print(f"Extracting embeddings (batch_size={batch_size})...")
    print("First batch will be slow (JIT compilation)...")
    embeddings = []
    num_batches = (len(all_audio) + batch_size - 1) // batch_size

    for i in tqdm(range(num_batches), desc="Extracting embeddings"):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, len(all_audio))
        batch = jnp.array(all_audio[start_idx:end_idx])
        lengths = jnp.array(all_lengths[start_idx:end_idx])

        batch_emb = extract_batch_embeddings(
            model,
            batch,
            lengths,
            pooling=pooling,
            feature_source=feature_source,
        )
        embeddings.append(np.array(batch_emb))

        if i == 0:
            print("JIT compilation done. Subsequent batches will be faster.")

    return np.concatenate(embeddings, axis=0)
