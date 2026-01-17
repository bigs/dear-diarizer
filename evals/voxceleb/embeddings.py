"""Extract embeddings from frozen WavLeJEPA checkpoint."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import equinox as eqx
import librosa
import numpy as np
from tqdm import tqdm

from wavlejepa.model import WavLeJEPA, WavLeJEPAConfig
from wavlejepa.training.checkpoint import WavLeJEPACheckpointer
from wavlejepa.training.config import TrainingConfig


def load_audio(path: Path, sample_rate: int = 16000) -> jnp.ndarray:
    """Load and resample audio file.

    Args:
        path: Audio file path
        sample_rate: Target sample rate

    Returns:
        Audio waveform as JAX array
    """
    audio, _ = librosa.load(path, sr=sample_rate, mono=True)
    return jnp.array(audio, dtype=jnp.float32)


def extract_utterance_embedding(
    model: WavLeJEPA,
    audio: jnp.ndarray,
) -> jnp.ndarray:
    """Extract utterance embedding from audio.

    Args:
        model: Frozen WavLeJEPA model (no gradients computed)
        audio: Audio waveform [T]

    Returns:
        Utterance embedding [768] (mean-pooled frame embeddings)
    """
    # Process: raw audio → frame embeddings → contextualized frames
    # No gradient computation - we're just doing inference
    frame_embeddings = model.waveform_encoder(audio)  # [N, 768]
    contextualized = model.context_encoder(frame_embeddings)  # [N, 768]

    # Mean pool over time to get utterance-level embedding
    utterance_emb = jnp.mean(contextualized, axis=0)  # [768]

    return utterance_emb


def extract_embeddings(
    checkpoint_path: Path,
    audio_paths: list[Path],
    batch_size: int = 32,
    sample_rate: int = 16000,
) -> np.ndarray:
    """Extract embeddings for a list of audio files.

    Args:
        checkpoint_path: Path to WavLeJEPA checkpoint directory
        audio_paths: List of audio file paths
        batch_size: Batch size (currently processes sequentially)
        sample_rate: Audio sample rate

    Returns:
        Embeddings array [num_utterances, 768]
    """
    checkpoint_path = Path(checkpoint_path)

    # Load configs
    training_config_path = checkpoint_path / "training_config.json"
    model_config_path = checkpoint_path / "model_config.json"

    # Reconstruct configs
    training_config = TrainingConfig.from_json(training_config_path)

    with open(model_config_path) as f:
        model_config_dict = json.load(f)
    model_config = WavLeJEPAConfig(**model_config_dict)

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

    # Extract embeddings
    embeddings = []
    for audio_path in tqdm(audio_paths, desc="Extracting embeddings"):
        audio = load_audio(audio_path, sample_rate=sample_rate)
        emb = extract_utterance_embedding(model, audio)
        embeddings.append(np.array(emb))

    return np.stack(embeddings)
