"""Linear probe evaluation on VoxCeleb1.

Evaluates WavLeJEPA representations by training a linear classifier
for speaker identification.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .data import load_voxceleb1_test
from .embeddings import extract_embeddings
from .linear_probe import train_linear_probe


def main():
    """Run VoxCeleb linear probe evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate WavLeJEPA with linear probe on VoxCeleb1",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to WavLeJEPA checkpoint (e.g., checkpoints-batch128-conservative/best)",
    )
    parser.add_argument(
        "--voxceleb-root",
        type=Path,
        required=True,
        help="Path to VoxCeleb1 dataset root",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding extraction",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/voxceleb_probe.json"),
        help="Output path for results JSON",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data for testing",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Audio sample rate",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=10.0,
        help="Maximum audio duration in seconds (longer files are cropped). "
             "Default 10s produces ~500 frames at 50Hz, safely under max_seq_len=1000.",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("VoxCeleb Linear Probe Evaluation")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"VoxCeleb root: {args.voxceleb_root}")
    print(f"Batch size: {args.batch_size}")
    print(f"Test size: {args.test_size}")
    print(f"Output: {args.output}")
    print()

    # Load VoxCeleb data
    print("Loading VoxCeleb1 test set...")
    audio_files = load_voxceleb1_test(args.voxceleb_root)
    print(f"Found {len(audio_files)} utterances from {len(set(y for _, y in audio_files))} speakers")
    print()

    # Extract file paths and labels
    file_paths = [f for f, _ in audio_files]
    labels = np.array([y for _, y in audio_files])

    # Extract embeddings
    print("Extracting embeddings from frozen WavLeJEPA encoder...")
    print(f"(Cropping audio to {args.max_duration}s to fit model's max_seq_len)")
    embeddings = extract_embeddings(
        checkpoint_path=args.checkpoint,
        audio_paths=file_paths,
        batch_size=args.batch_size,
        sample_rate=args.sample_rate,
        max_duration=args.max_duration,
    )
    print(f"Extracted embeddings shape: {embeddings.shape}")
    print()

    # Train linear probe
    print("Training linear probe...")
    results = train_linear_probe(
        embeddings=embeddings,
        labels=labels,
        test_size=args.test_size,
    )

    # Print results
    print()
    print("=" * 80)
    print("Results")
    print("=" * 80)
    print(f"Top-1 Accuracy: {results['top1_accuracy']:.2%}")
    if results['top5_accuracy'] is not None:
        print(f"Top-5 Accuracy: {results['top5_accuracy']:.2%}")
    print(f"Number of speakers: {results['num_classes']}")
    print(f"Train samples: {results['num_train']}")
    print(f"Test samples: {results['num_test']}")
    print()

    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "checkpoint": str(args.checkpoint),
        "voxceleb_root": str(args.voxceleb_root),
        "test_size": args.test_size,
        "results": results,
    }

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
