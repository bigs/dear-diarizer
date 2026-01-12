"""
Waveform Encoder for WavLeJEPA.

Modified Wav2Vec 2.0 feature encoder that converts raw 16kHz audio
to embeddings at 100Hz (10ms stride, ~15ms receptive field).
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import equinox as eqx
import librosa
import numpy as np
from jaxtyping import Array, Float, PRNGKeyArray

# Target sample rate for the encoder
TARGET_SR = 16000


# Wav2Vec 2.0 base config with last layer removed for 100Hz output
# Format: (out_channels, kernel_size, stride)
CONV_LAYERS = [
    (512, 10, 5),  # 16kHz -> 3200Hz
    (512, 3, 2),  # -> 1600Hz
    (512, 3, 2),  # -> 800Hz
    (512, 3, 2),  # -> 400Hz
    (512, 3, 2),  # -> 200Hz
    (512, 2, 2),  # -> 100Hz
]


class ConvBlock(eqx.Module):
    """Single convolutional block with GroupNorm and GELU."""

    conv: eqx.nn.Conv1d
    norm: eqx.nn.GroupNorm
    is_first: bool = eqx.field(static=True)

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        num_groups: int,
        *,
        is_first: bool = False,
        key: PRNGKeyArray,
    ):
        self.conv = eqx.nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            key=key,
        )
        self.norm = eqx.nn.GroupNorm(groups=num_groups, channels=out_channels)
        self.is_first = is_first

    def __call__(
        self, x: Float[Array, "channels time"]
    ) -> Float[Array, "channels time"]:
        x = self.conv(x)
        # GroupNorm expects (channels, ...) which we have
        x = self.norm(x)
        x = jax.nn.gelu(x)
        return x


class WaveformEncoder(eqx.Module):
    """
    Waveform encoder that converts raw audio to dense embeddings.

    Based on Wav2Vec 2.0 feature encoder with the last layer removed
    for finer-grained (100Hz) embeddings.

    Input: Raw waveform at 16kHz, shape (batch, time) or (time,)
    Output: Embeddings at 100Hz, shape (batch, frames, 768) or (frames, 768)
    """

    conv_blocks: list[ConvBlock]
    proj: eqx.nn.Linear
    embed_dim: int = eqx.field(static=True)

    def __init__(
        self,
        embed_dim: int = 768,
        num_groups: int = 32,
        *,
        key: PRNGKeyArray,
    ):
        """
        Args:
            embed_dim: Output embedding dimension (default 768 for ViT compatibility)
            num_groups: Number of groups for GroupNorm (default 32)
            key: JAX PRNG key
        """
        self.embed_dim = embed_dim

        keys = jax.random.split(key, len(CONV_LAYERS) + 1)

        # Build conv blocks
        blocks = []
        in_channels = 1  # Raw audio is mono
        for i, (out_channels, kernel_size, stride) in enumerate(CONV_LAYERS):
            block = ConvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                num_groups=num_groups,
                is_first=(i == 0),
                key=keys[i],
            )
            blocks.append(block)
            in_channels = out_channels

        self.conv_blocks = blocks

        # Project from conv output (512) to embed_dim (768)
        self.proj = eqx.nn.Linear(
            in_features=CONV_LAYERS[-1][0],  # 512
            out_features=embed_dim,
            key=keys[-1],
        )

    def __call__(
        self, x: Float[Array, "... time"]
    ) -> Float[Array, "... frames embed_dim"]:
        """
        Encode raw waveform to embeddings.

        Args:
            x: Raw waveform, shape (..., time) at 16kHz

        Returns:
            Embeddings, shape (..., frames, embed_dim) at 100Hz
        """
        # Handle batch dimension
        if x.ndim == 1:
            x = x[jnp.newaxis, :]  # Add batch dim
            squeeze_batch = True
        else:
            squeeze_batch = False

        # Reshape to (batch, 1, time) for conv1d - channels first
        x = x[:, jnp.newaxis, :]

        # Apply conv blocks (vmapped over batch)
        def apply_blocks(x_single):
            for block in self.conv_blocks:
                x_single = block(x_single)
            return x_single

        # x is (batch, 1, time), each x_single is (1, time)
        x = jax.vmap(apply_blocks)(x)  # (batch, 512, frames)

        # Transpose to (batch, frames, 512) for linear projection
        x = jnp.transpose(x, (0, 2, 1))

        # Project to embed_dim
        x = jax.vmap(jax.vmap(self.proj))(x)  # (batch, frames, embed_dim)

        if squeeze_batch:
            x = x[0]  # Remove batch dim

        return x

    @property
    def total_stride(self) -> int:
        """Total downsampling factor (160 for 100Hz at 16kHz)."""
        stride = 1
        for _, _, s in CONV_LAYERS:
            stride *= s
        return stride

    @property
    def receptive_field(self) -> int:
        """Receptive field in samples."""
        rf = CONV_LAYERS[0][1]  # First kernel size
        cumulative_stride = CONV_LAYERS[0][2]
        for _, kernel_size, stride in CONV_LAYERS[1:]:
            rf += (kernel_size - 1) * cumulative_stride
            cumulative_stride *= stride
        return rf

    def output_length(self, input_length: int) -> int:
        """Compute output length for a given input length."""
        length = input_length
        for _, kernel_size, stride in CONV_LAYERS:
            length = (length - kernel_size) // stride + 1
        return length


def load_audio(
    path: str | Path,
    *,
    target_sr: int = TARGET_SR,
    mono: bool = True,
    normalize: bool = True,
    duration: float | None = None,
    offset: float = 0.0,
) -> Float[Array, " time"]:
    """
    Load an audio file and prepare it for the WaveformEncoder.

    Handles stereo MP3s (and other formats) by:
    - Resampling to target sample rate (default 16kHz)
    - Converting stereo to mono by averaging channels
    - Mean-centering for equal loudness (per WavJEPA spec)

    Args:
        path: Path to audio file (mp3, wav, flac, etc.)
        target_sr: Target sample rate in Hz (default 16000)
        mono: If True, convert to mono by averaging channels
        normalize: If True, mean-center the audio
        duration: Duration in seconds to load (None for full file)
        offset: Start offset in seconds

    Returns:
        JAX array of shape (time,) with audio samples
    """
    # Load with librosa - handles resampling and format detection
    # mono=False to get original channels, we'll handle conversion ourselves
    audio, sr = librosa.load(
        path,
        sr=target_sr,
        mono=False,
        duration=duration,
        offset=offset,
    )

    # Handle stereo -> mono conversion
    if audio.ndim == 2:
        if mono:
            # Average channels for mono conversion
            audio = np.mean(audio, axis=0)
        else:
            raise ValueError(
                f"Audio has {audio.shape[0]} channels but mono=False. "
                "Set mono=True to convert to mono."
            )

    # Mean-center for equal loudness (per WavJEPA spec)
    if normalize:
        audio = audio - np.mean(audio)

    return jnp.array(audio, dtype=jnp.float32)


def load_audio_batch(
    paths: list[str | Path],
    *,
    target_sr: int = TARGET_SR,
    normalize: bool = True,
    duration: float | None = None,
    pad: bool = True,
) -> Float[Array, "batch time"]:
    """
    Load multiple audio files into a batched tensor.

    Args:
        paths: List of paths to audio files
        target_sr: Target sample rate in Hz (default 16000)
        normalize: If True, mean-center each audio clip
        duration: Duration in seconds to load (None for full file)
        pad: If True, pad shorter clips to match longest. If False,
             all clips must have same length.

    Returns:
        JAX array of shape (batch, time)
    """
    audios = [
        load_audio(p, target_sr=target_sr, normalize=normalize, duration=duration)
        for p in paths
    ]

    if not pad:
        lengths = [a.shape[0] for a in audios]
        if len(set(lengths)) > 1:
            raise ValueError(
                f"Audio clips have different lengths {lengths}. "
                "Set pad=True to pad to longest, or specify duration."
            )
        return jnp.stack(audios)

    # Pad to longest
    max_len = max(a.shape[0] for a in audios)
    padded = [
        jnp.pad(a, (0, max_len - a.shape[0])) if a.shape[0] < max_len else a
        for a in audios
    ]
    return jnp.stack(padded)


def random_crop(
    audio: Float[Array, " time"],
    crop_seconds: float,
    key: PRNGKeyArray,
    *,
    sample_rate: int = TARGET_SR,
) -> Float[Array, " crop_time"]:
    """
    Randomly crop a segment from audio (for training augmentation).

    Per WavJEPA: randomly sample 2-second sections from each clip.

    Args:
        audio: Audio tensor of shape (time,)
        crop_seconds: Duration of crop in seconds
        key: JAX PRNG key for random offset
        sample_rate: Sample rate of audio

    Returns:
        Cropped audio of shape (crop_samples,)
    """
    crop_samples = int(crop_seconds * sample_rate)
    total_samples = audio.shape[0]

    if total_samples <= crop_samples:
        # Pad if audio is shorter than crop
        return jnp.pad(audio, (0, crop_samples - total_samples))

    # Random start position
    max_start = total_samples - crop_samples
    start = jax.random.randint(key, (), 0, max_start + 1)

    return jax.lax.dynamic_slice(audio, (start,), (crop_samples,))
