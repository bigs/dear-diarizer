"""
Data pipeline for WavLeJEPA training.

Uses WebDataset to stream audio from object storage (S3/GCS/local).
"""

import io
from dataclasses import dataclass
from typing import Iterator, Callable

import jax
import jax.numpy as jnp
import numpy as np
import librosa
import webdataset as wds
from jaxtyping import Array, Float, PRNGKeyArray

from .waveform_encoder import TARGET_SR


@dataclass
class DataConfig:
    """Data pipeline configuration."""

    # Data source (supports s3://, gs://, or local paths)
    # Use brace expansion: "s3://bucket/shards-{000000..000999}.tar"
    shards_path: str

    # Audio parameters
    sample_rate: int = TARGET_SR  # 16kHz
    crop_duration: float = 2.0  # seconds

    # Batching
    batch_size: int = 32
    shuffle_buffer: int = 1000

    # Workers
    num_workers: int = 4


def load_audio_from_bytes(
    audio_bytes: bytes,
    sample_rate: int = TARGET_SR,
    normalize: bool = True,
) -> np.ndarray:
    """
    Load audio from bytes (MP3, WAV, FLAC, etc.) into numpy array.

    Args:
        audio_bytes: Raw audio file bytes
        sample_rate: Target sample rate (default 16kHz)
        normalize: If True, mean-center the audio

    Returns:
        Numpy array of shape (time,) with audio samples
    """
    # librosa can load from file-like objects
    audio, _ = librosa.load(
        io.BytesIO(audio_bytes),
        sr=sample_rate,
        mono=True,
    )

    if normalize:
        audio = audio - np.mean(audio)

    return audio.astype(np.float32)


def random_crop_np(
    audio: np.ndarray,
    crop_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Randomly crop a segment from audio.

    Args:
        audio: Audio array of shape (time,)
        crop_samples: Number of samples to crop
        rng: Numpy random generator

    Returns:
        Cropped audio of shape (crop_samples,)
    """
    if len(audio) <= crop_samples:
        # Pad if too short
        pad_amount = crop_samples - len(audio)
        return np.pad(audio, (0, pad_amount), mode="constant")

    max_start = len(audio) - crop_samples
    start = rng.integers(0, max_start + 1)
    return audio[start : start + crop_samples]


def create_dataset(
    config: DataConfig,
    seed: int = 42,
) -> wds.WebDataset:
    """
    Create a WebDataset pipeline for audio.

    Args:
        config: Data configuration
        seed: Random seed for shuffling

    Returns:
        WebDataset that yields processed audio samples
    """
    crop_samples = int(config.crop_duration * config.sample_rate)
    rng = np.random.default_rng(seed)

    def process_sample(sample: dict) -> dict:
        """Process a single sample from the tar."""
        # Find audio key (could be mp3, wav, flac, etc. - no leading dot in webdataset keys)
        audio_extensions = ("mp3", "wav", "flac", "ogg", "m4a", "opus", "webm")
        audio_key = None
        for key in sample:
            # Check both with and without dot (webdataset uses no dot)
            if key in audio_extensions or key.lstrip(".") in audio_extensions:
                audio_key = key
                break

        if audio_key is None:
            raise ValueError(f"No audio file found in sample: {list(sample.keys())}. Expected one of {audio_extensions}")

        # Load and process audio
        audio = load_audio_from_bytes(
            sample[audio_key],
            sample_rate=config.sample_rate,
        )

        # Random crop
        audio = random_crop_np(audio, crop_samples, rng)

        return {"audio": audio, "__key__": sample.get("__key__", "")}

    dataset = (
        wds.WebDataset(config.shards_path, shardshuffle=100)
        .shuffle(config.shuffle_buffer)
        .map(process_sample)
    )

    return dataset


def create_dataloader(
    config: DataConfig,
    seed: int = 42,
) -> Iterator[Float[Array, "batch time"]]:
    """
    Create an iterator that yields batched JAX arrays.

    Args:
        config: Data configuration
        seed: Random seed

    Yields:
        Batched audio arrays of shape (batch_size, crop_samples)
    """
    dataset = create_dataset(config, seed)
    crop_samples = int(config.crop_duration * config.sample_rate)

    batch = []
    for sample in dataset:
        batch.append(sample["audio"])

        if len(batch) >= config.batch_size:
            # Stack and convert to JAX
            batch_array = np.stack(batch[: config.batch_size])
            yield jnp.array(batch_array)
            batch = batch[config.batch_size :]


class AudioDataLoader:
    """
    Iterable dataloader that yields batched JAX arrays.

    Supports infinite iteration for training loops.
    """

    def __init__(
        self,
        config: DataConfig,
        seed: int = 42,
        infinite: bool = True,
    ):
        self.config = config
        self.seed = seed
        self.infinite = infinite
        self.crop_samples = int(config.crop_duration * config.sample_rate)

    def __iter__(self) -> Iterator[Float[Array, "batch time"]]:
        epoch = 0
        while True:
            # Create fresh dataset each epoch (for proper reshuffling)
            dataset = create_dataset(self.config, seed=self.seed + epoch)

            batch = []
            for sample in dataset:
                batch.append(sample["audio"])

                if len(batch) >= self.config.batch_size:
                    batch_array = np.stack(batch[: self.config.batch_size])
                    yield jnp.array(batch_array)
                    batch = batch[self.config.batch_size :]

            epoch += 1
            if not self.infinite:
                break


# =============================================================================
# Shard creation utilities
# =============================================================================


def create_shards(
    audio_paths: list[str],
    output_pattern: str,
    max_count: int = 1000,
    max_size: int = 1e9,  # 1GB
) -> int:
    """
    Create WebDataset shards from a list of audio files.

    Args:
        audio_paths: List of paths to audio files
        output_pattern: Output pattern like "shards/train-%06d.tar" or "s3://bucket/train-%06d.tar"
        max_count: Maximum samples per shard
        max_size: Maximum bytes per shard

    Returns:
        Number of shards created

    Example:
        >>> paths = glob.glob("/data/audio/*.mp3")
        >>> create_shards(paths, "s3://my-bucket/audio/train-%06d.tar")
    """
    from pathlib import Path

    with wds.ShardWriter(
        output_pattern,
        maxcount=max_count,
        maxsize=int(max_size),
    ) as sink:
        for path in audio_paths:
            path = Path(path)
            key = path.stem  # filename without extension

            # Determine extension
            ext = path.suffix.lower()
            if ext.startswith("."):
                ext = ext[1:]  # remove leading dot

            # Read file bytes
            with open(path, "rb") as f:
                audio_bytes = f.read()

            # Write to shard
            sink.write({
                "__key__": key,
                ext: audio_bytes,
            })

    # ShardWriter doesn't expose shard count, count from pattern
    return len(audio_paths) // max_count + (1 if len(audio_paths) % max_count else 0)


def create_shards_from_directory(
    input_dir: str,
    output_pattern: str,
    extensions: tuple[str, ...] = (".mp3", ".wav", ".flac", ".ogg"),
    max_count: int = 1000,
) -> int:
    """
    Create shards from all audio files in a directory.

    Args:
        input_dir: Directory containing audio files
        output_pattern: Output pattern for shards
        extensions: Audio file extensions to include
        max_count: Maximum samples per shard

    Returns:
        Number of shards created

    Example:
        >>> create_shards_from_directory(
        ...     "/data/audioset/",
        ...     "s3://my-bucket/audioset/train-%06d.tar",
        ... )
    """
    from pathlib import Path

    input_path = Path(input_dir)
    audio_paths = []
    for ext in extensions:
        audio_paths.extend(input_path.rglob(f"*{ext}"))

    audio_paths = [str(p) for p in sorted(audio_paths)]
    print(f"Found {len(audio_paths)} audio files")

    return create_shards(audio_paths, output_pattern, max_count=max_count)
