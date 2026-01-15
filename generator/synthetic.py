"""Synthetic multi-speaker test data generation.

Generates fake frame embeddings with clear cluster structure for:
- Unit testing energy functions
- Integration testing full generator
- Debugging training loop
- Visualizing attractor quality
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray


def make_synthetic_data(
    num_speakers: int,
    frames_per_speaker: int,
    embedding_dim: int = 768,
    noise_scale: float = 0.1,
    center_scale: float = 2.0,
    *,
    key: PRNGKeyArray,
) -> tuple[
    Float[Array, "num_frames embedding_dim"],
    Int[Array, " num_frames"],
    Float[Array, "num_speakers embedding_dim"],
]:
    """Generate synthetic multi-speaker frame embeddings.

    Creates `num_speakers` clusters in embedding space, each with
    `frames_per_speaker` frames. Frames are Gaussian around cluster centers.

    Args:
        num_speakers: Number of distinct speakers (clusters)
        frames_per_speaker: Number of frames per speaker
        embedding_dim: Dimension of embeddings (default 768 to match WavLeJEPA)
        noise_scale: Std dev of Gaussian noise around centers
        center_scale: Std dev of cluster centers from origin
        key: PRNG key

    Returns:
        frame_embeddings: [N, D] where N = num_speakers * frames_per_speaker
        speaker_labels: [N] ground truth speaker indices (for evaluation)
        speaker_centers: [num_speakers, D] true cluster centers (for evaluation)
    """
    k1, k2 = jax.random.split(key)

    N = num_speakers * frames_per_speaker

    # Generate cluster centers
    centers = jax.random.normal(k1, (num_speakers, embedding_dim)) * center_scale

    # Assign frames to speakers
    labels = jnp.repeat(jnp.arange(num_speakers), frames_per_speaker)  # [N]

    # Generate frames as center + noise
    noise = jax.random.normal(k2, (N, embedding_dim)) * noise_scale
    frames = centers[labels] + noise

    return frames, labels, centers


def make_variable_length_data(
    num_speakers: int,
    min_frames: int,
    max_frames: int,
    embedding_dim: int = 768,
    noise_scale: float = 0.1,
    center_scale: float = 2.0,
    *,
    key: PRNGKeyArray,
) -> tuple[
    Float[Array, "num_frames embedding_dim"],
    Int[Array, " num_frames"],
    Float[Array, "num_speakers embedding_dim"],
]:
    """Generate synthetic data with variable frames per speaker.

    More realistic than fixed frames per speaker.

    Args:
        num_speakers: Number of distinct speakers
        min_frames: Minimum frames per speaker
        max_frames: Maximum frames per speaker
        embedding_dim: Dimension of embeddings
        noise_scale: Std dev of Gaussian noise around centers
        center_scale: Std dev of cluster centers from origin
        key: PRNG key

    Returns:
        frame_embeddings: [N, D] where N varies
        speaker_labels: [N] ground truth speaker indices
        speaker_centers: [num_speakers, D] true cluster centers
    """
    k1, k2, k3 = jax.random.split(key, 3)

    # Generate cluster centers
    centers = jax.random.normal(k1, (num_speakers, embedding_dim)) * center_scale

    # Random frames per speaker
    frames_per_speaker = jax.random.randint(
        k2, (num_speakers,), minval=min_frames, maxval=max_frames + 1
    )
    total_frames = int(jnp.sum(frames_per_speaker))

    # Build labels array
    labels_list = []
    for i in range(num_speakers):
        labels_list.append(jnp.full((int(frames_per_speaker[i]),), i, dtype=jnp.int32))
    labels = jnp.concatenate(labels_list)

    # Generate frames
    noise = jax.random.normal(k3, (total_frames, embedding_dim)) * noise_scale
    frames = centers[labels] + noise

    return frames, labels, centers


def make_overlapping_speakers(
    num_speakers: int,
    frames_per_speaker: int,
    embedding_dim: int = 768,
    noise_scale: float = 0.1,
    center_scale: float = 2.0,
    overlap_ratio: float = 0.3,
    *,
    key: PRNGKeyArray,
) -> tuple[
    Float[Array, "num_frames embedding_dim"],
    Int[Array, " num_frames"],
    Float[Array, "num_speakers embedding_dim"],
]:
    """Generate data with some speakers closer together (harder case).

    Creates pairs of speakers that are closer than typical, simulating
    similar-sounding speakers that are harder to distinguish.

    Args:
        num_speakers: Number of distinct speakers (should be even for pairing)
        frames_per_speaker: Number of frames per speaker
        embedding_dim: Dimension of embeddings
        noise_scale: Std dev of Gaussian noise around centers
        center_scale: Std dev of cluster centers from origin
        overlap_ratio: How close paired speakers are (0=same, 1=independent)
        key: PRNG key

    Returns:
        frame_embeddings: [N, D]
        speaker_labels: [N]
        speaker_centers: [num_speakers, D]
    """
    k1, k2, k3 = jax.random.split(key, 3)

    N = num_speakers * frames_per_speaker

    # Generate base centers (half as many)
    num_pairs = num_speakers // 2
    remainder = num_speakers % 2

    base_centers = jax.random.normal(k1, (num_pairs, embedding_dim)) * center_scale

    # Create paired centers with offset
    offsets = jax.random.normal(k2, (num_pairs, embedding_dim)) * center_scale * overlap_ratio

    paired_centers = jnp.concatenate([
        base_centers,
        base_centers + offsets,
    ], axis=0)  # [num_pairs * 2, D]

    # Add any remaining unpaired speaker
    if remainder > 0:
        extra = jax.random.normal(k2, (1, embedding_dim)) * center_scale
        centers = jnp.concatenate([paired_centers, extra], axis=0)
    else:
        centers = paired_centers

    # Generate frames
    labels = jnp.repeat(jnp.arange(num_speakers), frames_per_speaker)
    noise = jax.random.normal(k3, (N, embedding_dim)) * noise_scale
    frames = centers[labels] + noise

    return frames, labels, centers
