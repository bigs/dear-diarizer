"""VoxCeleb1 speaker verification eval (EER calibration)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .embeddings import extract_embeddings
from .verification import (
    Trial,
    load_trials,
    unique_trial_files,
    l2_normalize,
    cosine_scores,
    compute_eer,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _manifest_path(cache_dir: Path) -> Path:
    return cache_dir / "manifest.json"


def _embeddings_path(cache_dir: Path) -> Path:
    return cache_dir / "embeddings.npy"


def _load_cache(
    cache_dir: Path,
    expected_files: list[Path],
    checkpoint: Path,
    sample_rate: int,
    max_duration: float,
    pooling: str,
    feature_source: str,
) -> np.ndarray | None:
    manifest_path = _manifest_path(cache_dir)
    embeddings_path = _embeddings_path(cache_dir)
    if not manifest_path.exists() or not embeddings_path.exists():
        return None

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    if manifest.get("checkpoint") != str(checkpoint):
        return None
    if manifest.get("sample_rate") != sample_rate:
        return None
    if manifest.get("max_duration") != max_duration:
        return None
    if manifest.get("pooling") != pooling:
        return None
    if manifest.get("feature_source") != feature_source:
        return None

    cached_files = [Path(p) for p in manifest.get("files", [])]
    if cached_files != expected_files:
        return None

    embeddings = np.load(embeddings_path)
    if embeddings.shape[0] != len(expected_files):
        return None

    return embeddings


def _save_cache(
    cache_dir: Path,
    files: list[Path],
    embeddings: np.ndarray,
    metadata: dict,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    _write_json(_manifest_path(cache_dir), {"files": [str(p) for p in files], **metadata})
    np.save(_embeddings_path(cache_dir), embeddings)


def _extract_or_load_embeddings(
    checkpoint: Path,
    files: list[Path],
    cache_dir: Path,
    sample_rate: int,
    max_duration: float,
    batch_size: int,
    pooling: str,
    feature_source: str,
    force: bool,
) -> np.ndarray:
    if not force:
        cached = _load_cache(
            cache_dir,
            files,
            checkpoint,
            sample_rate,
            max_duration,
            pooling,
            feature_source,
        )
        if cached is not None:
            print(f"Loaded cached embeddings from {cache_dir}")
            return cached

    print("Extracting embeddings from frozen WavLeJEPA encoder...")
    embeddings = extract_embeddings(
        checkpoint_path=checkpoint,
        audio_paths=files,
        batch_size=batch_size,
        sample_rate=sample_rate,
        max_duration=max_duration,
        pooling=pooling,
        feature_source=feature_source,
    )

    _save_cache(
        cache_dir,
        files,
        embeddings,
        metadata={
            "checkpoint": str(checkpoint),
            "sample_rate": sample_rate,
            "max_duration": max_duration,
            "batch_size": batch_size,
            "pooling": pooling,
            "feature_source": feature_source,
        },
    )
    return embeddings


def _labels_from_trials(trials: Iterable[Trial]) -> np.ndarray:
    return np.asarray([t.label for t in trials], dtype=np.int32)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VoxCeleb1 verification EER calibration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--voxceleb-root", type=Path, required=True)
    parser.add_argument(
        "--trials",
        type=Path,
        required=True,
        help="Path to VoxCeleb1 trial list (e.g., veri_test.txt)",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--max-duration", type=float, default=10.0)
    parser.add_argument(
        "--pooling",
        type=str,
        default="mean",
        choices=["mean", "meanstd"],
        help="Frame pooling strategy.",
    )
    parser.add_argument(
        "--feature-source",
        type=str,
        default="topk",
        choices=["topk", "context"],
        help="Frame feature source.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("results/voxceleb_verif_cache"),
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Ignore cache and recompute embeddings.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/voxceleb_verif.json"),
    )
    parser.add_argument(
        "--debug-stats",
        action="store_true",
        help="Print basic score/embedding statistics for sanity checking.",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("VoxCeleb1 Verification EER Calibration")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"VoxCeleb root: {args.voxceleb_root}")
    print(f"Trials: {args.trials}")
    print(f"Cache dir: {args.cache_dir}")
    print()

    trials = load_trials(args.trials, args.voxceleb_root)
    files = unique_trial_files(trials)
    print(f"Trials: {len(trials)}")
    print(f"Unique files: {len(files)}")

    embeddings = _extract_or_load_embeddings(
        checkpoint=args.checkpoint,
        files=files,
        cache_dir=args.cache_dir,
        sample_rate=args.sample_rate,
        max_duration=args.max_duration,
        batch_size=args.batch_size,
        pooling=args.pooling,
        feature_source=args.feature_source,
        force=args.force_recompute,
    )

    embeddings = l2_normalize(embeddings)
    file_index = {path: i for i, path in enumerate(files)}

    labels = _labels_from_trials(trials)
    scores = cosine_scores(embeddings, file_index, trials)
    eer, threshold = compute_eer(labels, scores)

    if args.debug_stats:
        pos_scores = scores[labels == 1]
        neg_scores = scores[labels == 0]
        emb_mean = embeddings.mean(axis=0)
        emb_std = embeddings.std(axis=0)
        print()
        print("Debug stats")
        print("-" * 80)
        print(f"Pos scores: mean={pos_scores.mean():.4f} std={pos_scores.std():.4f}")
        print(f"Neg scores: mean={neg_scores.mean():.4f} std={neg_scores.std():.4f}")
        print(f"Embedding std (mean over dims): {emb_std.mean():.6f}")
        print(f"Embedding mean norm: {np.linalg.norm(emb_mean):.6f}")

    results = {
        "eer": eer,
        "threshold": threshold,
        "num_trials": int(labels.shape[0]),
        "num_files": int(len(files)),
    }

    payload = {
        "checkpoint": str(args.checkpoint),
        "voxceleb_root": str(args.voxceleb_root),
        "trials": str(args.trials),
        "config": {
            "sample_rate": args.sample_rate,
            "max_duration": args.max_duration,
            "batch_size": args.batch_size,
            "cache_dir": str(args.cache_dir),
            "pooling": args.pooling,
            "feature_source": args.feature_source,
        },
        "results": results,
    }
    _write_json(args.output, payload)

    print()
    print("Results")
    print("-" * 80)
    print(f"EER: {eer:.4%}")
    print(f"Threshold: {threshold:.6f}")
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
