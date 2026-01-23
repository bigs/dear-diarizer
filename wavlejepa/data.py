"""
Data pipeline for WavLeJEPA training.

Uses WebDataset to stream audio from object storage (S3/GCS/local).
"""

import io
import os
import queue
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Callable

import jax.numpy as jnp
import numpy as np
import librosa
import webdataset as wds
from jaxtyping import Array, Float

from .waveform_encoder import TARGET_SR


@contextmanager
def suppress_stderr():
    """Suppress stderr (for noisy libmpg123 warnings)."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stderr = os.dup(2)
    os.dup2(devnull, 2)
    try:
        yield
    finally:
        os.dup2(old_stderr, 2)
        os.close(devnull)
        os.close(old_stderr)


@dataclass
class DataConfig:
    """Data pipeline configuration."""

    # Data source - one of these must be specified:
    # WebDataset: supports s3://, gs://, or local paths
    # Use brace expansion: "s3://bucket/shards-{000000..000999}.tar"
    shards_path: str | None = None

    # HuggingFace datasets: e.g., "agkphysics/AudioSet"
    hf_dataset: str | None = None
    hf_subset: str = "unbalanced"  # Dataset config/subset
    hf_split: str = "train"  # Dataset split

    # Audio parameters
    sample_rate: int = TARGET_SR  # 16kHz
    crop_duration: float = 2.0  # seconds

    # Multiple crops per audio file (like WavJEPA's samples_per_audio)
    # Extracts N random crops from each audio before moving to next file
    # With crops_per_audio=8 and batch_size=384, each batch contains
    # 48 unique audio files × 8 crops each = 384 samples
    crops_per_audio: int = 1

    # Batching
    batch_size: int = 32
    shuffle_buffer: int = 1000

    # Workers
    num_workers: int = 4
    prefetch_batches: int = 2


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
    # suppress_stderr silences noisy libmpg123 warnings
    with suppress_stderr():
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


_THREAD_LOCAL = threading.local()


def _get_thread_rng(seed: int) -> np.random.Generator:
    """Get a per-thread RNG seeded for the current epoch."""
    rng = getattr(_THREAD_LOCAL, "rng", None)
    if rng is None or getattr(_THREAD_LOCAL, "seed", None) != seed:
        thread_id = threading.get_ident() & 0xFFFFFFFF
        _THREAD_LOCAL.rng = np.random.default_rng((seed + thread_id) % (2**32))
        _THREAD_LOCAL.seed = seed
    return _THREAD_LOCAL.rng


def _parallel_map(
    fn: Callable[[dict], dict],
    iterable: Iterator[dict],
    num_workers: int,
    max_queue: int | None = None,
) -> Iterator[dict]:
    if num_workers <= 1:
        yield from map(fn, iterable)
        return

    if max_queue is None:
        max_queue = num_workers * 2

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        it = iter(iterable)
        futures = deque()

        for _ in range(max_queue):
            try:
                item = next(it)
            except StopIteration:
                break
            futures.append(executor.submit(fn, item))

        while futures:
            future = futures.popleft()
            yield future.result()
            try:
                item = next(it)
            except StopIteration:
                continue
            futures.append(executor.submit(fn, item))


def _prefetch_iterator(
    iterable: Iterator[np.ndarray],
    max_prefetch: int,
) -> Iterator[np.ndarray]:
    if max_prefetch <= 0:
        yield from iterable
        return

    q: queue.Queue[object] = queue.Queue(maxsize=max_prefetch)
    sentinel = object()
    error: list[BaseException] = []

    def worker() -> None:
        try:
            for item in iterable:
                q.put(item)
        except BaseException as exc:
            error.append(exc)
        finally:
            q.put(sentinel)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    while True:
        item = q.get()
        if item is sentinel:
            break
        yield item

    if error:
        raise error[0]


def _batch_samples(
    samples: Iterator[dict],
    batch_size: int,
) -> Iterator[np.ndarray]:
    batch: list[np.ndarray] = []
    for sample in samples:
        batch.append(sample["audio"])
        if len(batch) >= batch_size:
            batch_array = np.stack(batch[:batch_size])
            yield batch_array
            batch = batch[batch_size:]


def _expand_crops(
    samples: Iterator[dict],
    crops_per_audio: int,
    crop_samples: int,
    seed: int,
) -> Iterator[dict]:
    """
    Expand each audio into N random crops.

    Args:
        samples: Iterator yielding dicts with "audio_full" (full decoded audio)
        crops_per_audio: Number of crops to extract from each audio
        crop_samples: Number of samples per crop
        seed: Base seed for deterministic RNG

    Yields:
        Dicts with "audio" (cropped) and "__key__" (with _cropN suffix)
    """
    if crops_per_audio <= 1:
        yield from samples
        return

    for sample in samples:
        full_audio = sample["audio_full"]
        key = sample["__key__"]

        # Deterministic RNG per audio file for reproducibility
        rng = np.random.default_rng((seed + hash(key)) % (2**32))

        for i in range(crops_per_audio):
            crop = random_crop_np(full_audio, crop_samples, rng)
            yield {"audio": crop, "__key__": f"{key}_crop{i}"}


def _create_webdataset(
    config: DataConfig,
    seed: int = 42,
) -> Iterator[dict]:
    """
    Create an iterator of processed audio samples from WebDataset shards.

    Args:
        config: Data configuration
        seed: Random seed for shuffling/cropping

    Returns:
        Iterator that yields processed audio samples
    """
    crop_samples = int(config.crop_duration * config.sample_rate)
    multi_crop = config.crops_per_audio > 1

    def process_sample(sample: dict) -> dict:
        """Process a single sample from the tar."""
        rng = _get_thread_rng(seed)
        # Check for pre-decoded tensor first (fast path)
        if "npy" in sample:
            audio = np.load(io.BytesIO(sample["npy"]))
            if multi_crop:
                # Return full audio for later crop expansion
                return {"audio_full": audio, "__key__": sample.get("__key__", "")}
            # Still need random crop since chunks are typically 30s
            audio = random_crop_np(audio, crop_samples, rng)
            return {"audio": audio, "__key__": sample.get("__key__", "")}

        # Fall back to MP3/audio decoding (backward compatible)
        audio_extensions = ("mp3", "wav", "flac", "ogg", "m4a", "opus", "webm")
        audio_key = None
        for key in sample:
            # Check both with and without dot (webdataset uses no dot)
            if key in audio_extensions or key.lstrip(".") in audio_extensions:
                audio_key = key
                break

        if audio_key is None:
            raise ValueError(f"No audio file found in sample: {list(sample.keys())}. Expected one of {audio_extensions} or 'npy'")

        # Load and process audio
        audio = load_audio_from_bytes(
            sample[audio_key],
            sample_rate=config.sample_rate,
        )

        if multi_crop:
            # Return full audio for later crop expansion
            return {"audio_full": audio, "__key__": sample.get("__key__", "")}

        # Random crop
        audio = random_crop_np(audio, crop_samples, rng)

        return {"audio": audio, "__key__": sample.get("__key__", "")}

    dataset = (
        wds.WebDataset(config.shards_path, shardshuffle=100)
        .shuffle(config.shuffle_buffer)
    )

    if config.num_workers > 1:
        samples = _parallel_map(process_sample, dataset, config.num_workers)
    else:
        samples = map(process_sample, dataset)

    # Expand each audio into multiple crops if configured
    if multi_crop:
        samples = _expand_crops(samples, config.crops_per_audio, crop_samples, seed)

    return samples


def _create_hf_dataset(
    config: DataConfig,
    seed: int = 42,
) -> Iterator[dict]:
    """
    Create an iterator of processed audio samples from HuggingFace datasets.

    Streams audio from HF without downloading the full dataset.

    Args:
        config: Data configuration with hf_dataset specified
        seed: Random seed for shuffling/cropping

    Returns:
        Iterator that yields processed audio samples
    """
    from datasets import load_dataset, Audio, Features, Value, Sequence

    crop_samples = int(config.crop_duration * config.sample_rate)
    multi_crop = config.crops_per_audio > 1

    # Known dataset schemas with audio decoding disabled
    # This avoids the torchcodec dependency by getting raw bytes
    KNOWN_FEATURES = {
        "agkphysics/AudioSet": Features({
            "video_id": Value("string"),
            "audio": Audio(decode=False),
            "labels": Sequence(Value("string")),
            "human_labels": Sequence(Value("string")),
        }),
    }

    features = KNOWN_FEATURES.get(config.hf_dataset)

    # Load dataset in streaming mode
    ds = load_dataset(
        config.hf_dataset,
        config.hf_subset,
        split=config.hf_split,
        streaming=True,
        features=features,  # None for unknown datasets
    )

    # For unknown datasets, try to disable audio decoding via cast_column
    if features is None:
        try:
            ds = ds.cast_column("audio", Audio(decode=False))
        except (ValueError, KeyError):
            pass  # Column doesn't exist or can't be cast

    # Shuffle with buffer
    ds = ds.shuffle(seed=seed, buffer_size=config.shuffle_buffer)

    def process_hf_sample(sample: dict) -> dict:
        """Process a single sample from HuggingFace."""
        rng = _get_thread_rng(seed)

        # Decode audio bytes (FLAC for AudioSet)
        audio_bytes = sample["audio"]["bytes"]
        audio = load_audio_from_bytes(
            audio_bytes,
            sample_rate=config.sample_rate,
        )

        # Use video_id as key (AudioSet-specific, fallback to generic)
        key = sample.get("video_id", sample.get("id", ""))

        if multi_crop:
            # Return full audio for later crop expansion
            return {"audio_full": audio, "__key__": key}

        # Random crop (AudioSet clips are 10s, we want 2s)
        audio = random_crop_np(audio, crop_samples, rng)

        return {"audio": audio, "__key__": key}

    if config.num_workers > 1:
        samples = _parallel_map(process_hf_sample, ds, config.num_workers)
    else:
        samples = map(process_hf_sample, ds)

    # Expand each audio into multiple crops if configured
    if multi_crop:
        samples = _expand_crops(samples, config.crops_per_audio, crop_samples, seed)

    return samples


def create_dataset(
    config: DataConfig,
    seed: int = 42,
) -> Iterator[dict]:
    """
    Create an iterator of processed audio samples.

    Dispatches to WebDataset or HuggingFace datasets based on config.

    Args:
        config: Data configuration
        seed: Random seed for shuffling/cropping

    Returns:
        Iterator that yields processed audio samples
    """
    if config.hf_dataset:
        return _create_hf_dataset(config, seed)
    elif config.shards_path:
        return _create_webdataset(config, seed)
    else:
        raise ValueError("Must specify either shards_path or hf_dataset in DataConfig")


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
    samples = create_dataset(config, seed)
    batch_iter = _batch_samples(samples, config.batch_size)
    batch_iter = _prefetch_iterator(batch_iter, config.prefetch_batches)

    for batch_array in batch_iter:
        yield jnp.array(batch_array)


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
            samples = create_dataset(self.config, seed=self.seed + epoch)
            batch_iter = _batch_samples(samples, self.config.batch_size)
            batch_iter = _prefetch_iterator(batch_iter, self.config.prefetch_batches)

            for batch_array in batch_iter:
                yield jnp.array(batch_array)

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
