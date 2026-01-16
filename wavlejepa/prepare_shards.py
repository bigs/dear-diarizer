"""
Preprocess audio files into tensor shards for fast training.

Converts MP3/audio files to pre-decoded numpy arrays stored in WebDataset
shards, eliminating the MP3 decoding bottleneck during training.

Usage:
    python -m wavlejepa.prepare_shards \
        --input downloads/ \
        --output shards/train-%06d.tar \
        --chunk-duration 30.0 \
        --dtype float16
"""

import argparse
import gc
import io
import os
import sys
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Suppress librosa/libmpg123 warnings
import warnings
warnings.filterwarnings("ignore")


def suppress_stderr():
    """Context manager to suppress stderr (for libmpg123 warnings)."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stderr = os.dup(2)
    os.dup2(devnull, 2)
    try:
        yield
    finally:
        os.dup2(old_stderr, 2)
        os.close(devnull)
        os.close(old_stderr)


def load_audio(path: Path, sample_rate: int) -> np.ndarray:
    """Load and resample audio file."""
    import librosa

    # Suppress libmpg123 stderr warnings
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stderr = os.dup(2)
    os.dup2(devnull, 2)
    try:
        audio, _ = librosa.load(path, sr=sample_rate, mono=True)
    finally:
        os.dup2(old_stderr, 2)
        os.close(devnull)
        os.close(old_stderr)

    return audio


def split_into_chunks(
    audio: np.ndarray,
    chunk_samples: int,
    min_chunk_samples: int,
) -> list[np.ndarray]:
    """Split audio into fixed-size chunks."""
    chunks = []
    for start in range(0, len(audio), chunk_samples):
        chunk = audio[start : start + chunk_samples]

        # Discard chunks shorter than minimum
        if len(chunk) < min_chunk_samples:
            continue

        # Pad final chunk if needed (between min and full size)
        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)), mode="constant")

        chunks.append(chunk)

    return chunks


def process_file(
    path: Path,
    sample_rate: int,
    chunk_samples: int,
    min_chunk_samples: int,
    dtype: np.dtype,
) -> list[tuple[str, bytes]]:
    """
    Process a single audio file into chunks.

    Returns list of (key, npy_bytes) tuples.
    """
    try:
        audio = load_audio(path, sample_rate)
        chunks = split_into_chunks(audio, chunk_samples, min_chunk_samples)

        # Free the full audio array immediately
        del audio

        results = []
        for i, chunk in enumerate(chunks):
            key = f"{path.stem}_{i:04d}"
            chunk_typed = chunk.astype(dtype)

            # Serialize to bytes
            buf = io.BytesIO()
            np.save(buf, chunk_typed)
            npy_bytes = buf.getvalue()
            buf.close()

            results.append((key, npy_bytes))

            # Cleanup intermediate arrays
            del chunk_typed

        # Free the chunks list
        del chunks

        # Force garbage collection in worker before returning
        gc.collect()

        return results
    except Exception as e:
        tqdm.write(f"Warning: Failed to process {path}: {e}")
        return []


def find_audio_files(input_dir: Path, extensions: tuple[str, ...]) -> list[Path]:
    """Find all audio files in directory."""
    files = []
    for ext in extensions:
        files.extend(input_dir.glob(f"*{ext}"))
        files.extend(input_dir.glob(f"*{ext.upper()}"))
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess audio files into tensor shards",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Input directory containing audio files",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Output pattern for shards (e.g., shards/train-%%06d.tar)",
    )
    parser.add_argument(
        "--chunk-duration",
        type=float,
        default=30.0,
        help="Duration of each chunk in seconds",
    )
    parser.add_argument(
        "--min-chunk-duration",
        type=float,
        default=10.0,
        help="Minimum chunk duration (shorter chunks are discarded)",
    )
    parser.add_argument(
        "--dtype",
        choices=["float16", "float32"],
        default="float16",
        help="Output dtype for audio tensors",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Target sample rate",
    )
    parser.add_argument(
        "--max-shard-size",
        type=float,
        default=1e9,
        help="Maximum shard size in bytes",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: cpu_count - 1)",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".mp3", ".wav", ".flac", ".ogg", ".m4a"],
        help="Audio file extensions to process",
    )

    args = parser.parse_args()

    # Import webdataset here to avoid slow import at startup
    import webdataset as wds

    # Validate input
    if not args.input.is_dir():
        tqdm.write(f"Error: Input directory does not exist: {args.input}")
        sys.exit(1)

    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Find audio files
    audio_files = find_audio_files(args.input, tuple(args.extensions))
    if not audio_files:
        tqdm.write(f"Error: No audio files found in {args.input}")
        sys.exit(1)

    tqdm.write(f"Found {len(audio_files)} audio files")

    # Calculate chunk sizes
    chunk_samples = int(args.chunk_duration * args.sample_rate)
    min_chunk_samples = int(args.min_chunk_duration * args.sample_rate)
    dtype = np.float16 if args.dtype == "float16" else np.float32

    tqdm.write(f"Chunk size: {chunk_samples} samples ({args.chunk_duration}s)")
    tqdm.write(f"Min chunk size: {min_chunk_samples} samples ({args.min_chunk_duration}s)")
    tqdm.write(f"Output dtype: {dtype}")

    # Set up worker pool - conservative settings to manage memory
    num_workers = args.workers or max(1, min(4, cpu_count() - 1))
    tqdm.write(f"Using {num_workers} workers")

    # Create process function with fixed args
    process_fn = partial(
        process_file,
        sample_rate=args.sample_rate,
        chunk_samples=chunk_samples,
        min_chunk_samples=min_chunk_samples,
        dtype=dtype,
    )

    # Process files and write to shards
    total_chunks = 0
    total_bytes = 0

    with wds.ShardWriter(
        str(args.output),
        maxsize=int(args.max_shard_size),
    ) as sink:
        # maxtasksperchild=5 restarts workers frequently to free leaked memory
        with Pool(num_workers, maxtasksperchild=5) as pool:
            for chunks in tqdm(
                pool.imap_unordered(process_fn, audio_files),
                total=len(audio_files),
                desc="Processing",
            ):
                for key, npy_bytes in chunks:
                    sink.write({
                        "__key__": key,
                        "npy": npy_bytes,
                    })
                    total_chunks += 1
                    total_bytes += len(npy_bytes)

                # Explicit cleanup after each file
                del chunks
                gc.collect()

    tqdm.write(f"\nDone!")
    tqdm.write(f"Total chunks: {total_chunks}")
    tqdm.write(f"Total size: {total_bytes / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
