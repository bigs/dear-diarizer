"""Run VoxCeleb verification + collapse diagnostics and write a combined JSON."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from .collapse_check import (
    _collect_samples,
    _cosine_stats,
    _covariance_stats,
    _load_model,
)
from .verify import _extract_or_load_embeddings
from .verification import (
    Trial,
    compute_eer,
    cosine_scores,
    l2_normalize,
    load_trials,
    unique_trial_files,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _labels_from_trials(trials: list[Trial]) -> np.ndarray:
    return np.asarray([t.label for t in trials], dtype=np.int32)


def _sample_wavs(root: Path, num_files: int, seed: int) -> list[Path]:
    if root.name != "wav":
        root = root / "wav"

    all_files = sorted(root.rglob("*.wav"))
    if not all_files:
        raise FileNotFoundError(f"No wav files found under {root}")

    if num_files >= len(all_files):
        return all_files

    rng = random.Random(seed)
    return rng.sample(all_files, num_files)


def _run_collapse_suite(
    checkpoint: Path,
    voxceleb_root: Path,
    *,
    num_files: int,
    frames_per_file: int,
    num_pairs: int,
    max_cov_samples: int,
    batch_size: int,
    sample_rate: int,
    max_duration: float,
    seed: int,
    include_project: bool,
) -> dict:
    sample_files = _sample_wavs(voxceleb_root, num_files, seed)
    model = _load_model(checkpoint)

    configs = [
        ("topk", "topk", False),
        ("context", "context", False),
    ]
    if include_project:
        configs += [
            ("topk_projected", "topk", True),
            ("context_projected", "context", True),
        ]

    collapse_results = {}
    for name, feature_source, project in configs:
        frames, file_ids = _collect_samples(
            model=model,
            audio_paths=sample_files,
            sample_rate=sample_rate,
            max_duration=max_duration,
            batch_size=batch_size,
            frames_per_file=frames_per_file,
            feature_source=feature_source,
            project=project,
            seed=seed,
        )
        norms = np.linalg.norm(frames, axis=1)
        collapse_results[name] = {
            "num_samples": int(frames.shape[0]),
            "embedding_dim": int(frames.shape[1]),
            "norm_mean": float(norms.mean()),
            "norm_std": float(norms.std()),
            "cosine_stats": _cosine_stats(frames, file_ids, num_pairs, seed),
            "covariance_stats": _covariance_stats(frames, max_cov_samples, seed),
        }

    return collapse_results


def _run_verification(
    checkpoint: Path,
    voxceleb_root: Path,
    trials_path: Path,
    *,
    batch_size: int,
    sample_rate: int,
    max_duration: float,
    pooling: str,
    feature_source: str,
    cache_dir: Path,
    force_recompute: bool,
    debug_stats: bool,
) -> dict:
    trials = load_trials(trials_path, voxceleb_root)
    files = unique_trial_files(trials)

    embeddings = _extract_or_load_embeddings(
        checkpoint=checkpoint,
        files=files,
        cache_dir=cache_dir,
        sample_rate=sample_rate,
        max_duration=max_duration,
        batch_size=batch_size,
        pooling=pooling,
        feature_source=feature_source,
        force=force_recompute,
    )

    embeddings = l2_normalize(embeddings)
    file_index = {path: i for i, path in enumerate(files)}
    labels = _labels_from_trials(trials)
    scores = cosine_scores(embeddings, file_index, trials)
    eer, threshold = compute_eer(labels, scores)

    result = {
        "eer": float(eer),
        "threshold": float(threshold),
        "num_trials": int(labels.shape[0]),
        "num_files": int(len(files)),
    }

    if debug_stats:
        pos_scores = scores[labels == 1]
        neg_scores = scores[labels == 0]
        emb_mean = embeddings.mean(axis=0)
        emb_std = embeddings.std(axis=0)
        result["debug"] = {
            "pos_scores_mean": float(pos_scores.mean()),
            "pos_scores_std": float(pos_scores.std()),
            "neg_scores_mean": float(neg_scores.mean()),
            "neg_scores_std": float(neg_scores.std()),
            "embedding_std_mean": float(emb_std.mean()),
            "embedding_mean_norm": float(np.linalg.norm(emb_mean)),
        }

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run VoxCeleb verification + collapse diagnostics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--voxceleb-root", type=Path, required=True)
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/voxceleb_compare.json"))

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--max-duration", type=float, default=10.0)
    parser.add_argument(
        "--pooling",
        type=str,
        default="mean",
        choices=["mean", "meanstd"],
    )
    parser.add_argument(
        "--feature-source",
        type=str,
        default="topk",
        choices=["topk", "context"],
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("results/voxceleb_verif_cache"),
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Ignore cached embeddings.",
    )
    parser.add_argument(
        "--debug-stats",
        action="store_true",
        help="Include embedding score stats in output.",
    )

    parser.add_argument("--collapse-num-files", type=int, default=100)
    parser.add_argument("--collapse-frames-per-file", type=int, default=50)
    parser.add_argument("--collapse-num-pairs", type=int, default=50000)
    parser.add_argument("--collapse-max-cov-samples", type=int, default=5000)
    parser.add_argument("--collapse-batch-size", type=int, default=8)
    parser.add_argument(
        "--collapse-include-project",
        action="store_true",
        help="Include projector-space collapse metrics.",
    )
    parser.add_argument("--collapse-seed", type=int, default=0)

    args = parser.parse_args()

    print("=" * 80)
    print("VoxCeleb Verification + Collapse Diagnostics")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"VoxCeleb root: {args.voxceleb_root}")
    print(f"Trials: {args.trials}")
    print(f"Output: {args.output}")
    print()

    verification = _run_verification(
        checkpoint=args.checkpoint,
        voxceleb_root=args.voxceleb_root,
        trials_path=args.trials,
        batch_size=args.batch_size,
        sample_rate=args.sample_rate,
        max_duration=args.max_duration,
        pooling=args.pooling,
        feature_source=args.feature_source,
        cache_dir=args.cache_dir,
        force_recompute=args.force_recompute,
        debug_stats=args.debug_stats,
    )

    collapse = _run_collapse_suite(
        checkpoint=args.checkpoint,
        voxceleb_root=args.voxceleb_root,
        num_files=args.collapse_num_files,
        frames_per_file=args.collapse_frames_per_file,
        num_pairs=args.collapse_num_pairs,
        max_cov_samples=args.collapse_max_cov_samples,
        batch_size=args.collapse_batch_size,
        sample_rate=args.sample_rate,
        max_duration=args.max_duration,
        seed=args.collapse_seed,
        include_project=args.collapse_include_project,
    )

    payload = {
        "checkpoint": str(args.checkpoint),
        "voxceleb_root": str(args.voxceleb_root),
        "trials": str(args.trials),
        "verification_config": {
            "sample_rate": args.sample_rate,
            "max_duration": args.max_duration,
            "batch_size": args.batch_size,
            "pooling": args.pooling,
            "feature_source": args.feature_source,
            "cache_dir": str(args.cache_dir),
            "force_recompute": args.force_recompute,
        },
        "collapse_config": {
            "num_files": args.collapse_num_files,
            "frames_per_file": args.collapse_frames_per_file,
            "num_pairs": args.collapse_num_pairs,
            "max_cov_samples": args.collapse_max_cov_samples,
            "batch_size": args.collapse_batch_size,
            "sample_rate": args.sample_rate,
            "max_duration": args.max_duration,
            "seed": args.collapse_seed,
            "include_project": args.collapse_include_project,
        },
        "results": {
            "verification": verification,
            "collapse": collapse,
        },
    }

    _write_json(args.output, payload)
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
