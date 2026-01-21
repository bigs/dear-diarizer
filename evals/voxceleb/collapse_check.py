"""Embedding collapse diagnostics for WavLeJEPA."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np

from wavlejepa.model import WavLeJEPA, WavLeJEPAConfig
from wavlejepa.training.checkpoint import WavLeJEPACheckpointer
from wavlejepa.training.config import TrainingConfig

from .embeddings import load_audio_padded


@eqx.filter_jit
def _extract_topk(model: WavLeJEPA, audio_batch: jnp.ndarray) -> jnp.ndarray:
    return jax.vmap(lambda audio: model.extract_features(audio))(audio_batch)


@eqx.filter_jit
def _extract_context(model: WavLeJEPA, audio_batch: jnp.ndarray) -> jnp.ndarray:
    def _single(audio: jnp.ndarray) -> jnp.ndarray:
        frames = model.waveform_encoder(audio)
        return model.context_encoder(frames, inference=True)

    return jax.vmap(_single)(audio_batch)


@eqx.filter_jit
def _project_frames(model: WavLeJEPA, frames: jnp.ndarray) -> jnp.ndarray:
    return jax.vmap(lambda seq: jax.vmap(model.projector)(seq))(frames)


def _load_model(checkpoint_path: Path) -> WavLeJEPA:
    checkpoint_path = Path(checkpoint_path)
    training_config_path = checkpoint_path / "training_config.json"
    model_config_path = checkpoint_path / "model_config.json"

    training_config = TrainingConfig.from_json(training_config_path)
    with model_config_path.open("r", encoding="utf-8") as f:
        model_config = WavLeJEPAConfig(**json.load(f))

    checkpointer = WavLeJEPACheckpointer(
        config=training_config.checkpoint,
        training_config=training_config,
        model_config=model_config,
    )
    result = checkpointer.restore_best(key=jax.random.key(0))
    if result is None:
        raise ValueError(f"No checkpoint found at {checkpoint_path}")
    state, _ = result
    return state.model


def _collect_samples(
    model: WavLeJEPA,
    audio_paths: list[Path],
    sample_rate: int,
    max_duration: float,
    batch_size: int,
    frames_per_file: int,
    feature_source: str,
    project: bool,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    all_frames = []
    file_ids = []

    num_batches = (len(audio_paths) + batch_size - 1) // batch_size
    for i in range(num_batches):
        start = i * batch_size
        end = min(start + batch_size, len(audio_paths))
        batch_paths = audio_paths[start:end]

        audio = []
        lengths = []
        for path in batch_paths:
            wav, length = load_audio_padded(
                path, sample_rate=sample_rate, max_duration=max_duration
            )
            audio.append(wav)
            lengths.append(length)

        audio_batch = jnp.array(np.stack(audio))
        lengths = np.asarray(lengths, dtype=np.int32)

        if feature_source == "topk":
            frames = _extract_topk(model, audio_batch)
        elif feature_source == "context":
            frames = _extract_context(model, audio_batch)
        else:
            raise ValueError(f"Unsupported feature_source: {feature_source}")

        if project:
            frames = _project_frames(model, frames)

        frames = np.array(frames)
        for j, length in enumerate(lengths):
            valid_frames = int(model.waveform_encoder.output_length(length))
            valid_frames = max(1, min(valid_frames, frames.shape[1]))
            file_frames = frames[j, :valid_frames]
            if frames_per_file < valid_frames:
                idx = rng.choice(valid_frames, size=frames_per_file, replace=False)
                file_frames = file_frames[idx]
            all_frames.append(file_frames)
            file_ids.append(np.full(file_frames.shape[0], start + j, dtype=np.int32))

    all_frames = np.concatenate(all_frames, axis=0)
    file_ids = np.concatenate(file_ids, axis=0)
    return all_frames, file_ids


def _cosine_stats(frames: np.ndarray, file_ids: np.ndarray, num_pairs: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    norms = np.linalg.norm(frames, axis=1, keepdims=True)
    frames = frames / np.maximum(norms, 1e-12)

    n = frames.shape[0]
    idx1 = rng.integers(0, n, size=num_pairs)
    idx2 = rng.integers(0, n, size=num_pairs)
    cos = np.sum(frames[idx1] * frames[idx2], axis=1)
    same = file_ids[idx1] == file_ids[idx2]

    return {
        "cosine_mean": float(cos.mean()),
        "cosine_std": float(cos.std()),
        "within_mean": float(cos[same].mean()) if np.any(same) else None,
        "within_std": float(cos[same].std()) if np.any(same) else None,
        "between_mean": float(cos[~same].mean()) if np.any(~same) else None,
        "between_std": float(cos[~same].std()) if np.any(~same) else None,
    }


def _covariance_stats(frames: np.ndarray, max_samples: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    if frames.shape[0] > max_samples:
        idx = rng.choice(frames.shape[0], size=max_samples, replace=False)
        frames = frames[idx]

    mean = frames.mean(axis=0, keepdims=True)
    centered = frames - mean
    cov = centered.T @ centered / max(1, frames.shape[0])
    vals = np.linalg.eigvalsh(cov)
    vals = np.maximum(vals, 0)
    total = vals.sum()
    if total <= 0:
        return {"effective_rank": 0.0, "top1_ratio": 0.0}
    p = vals / total
    entropy = -np.sum(p * np.log(p + 1e-12))
    effective_rank = float(np.exp(entropy))
    top1_ratio = float(vals[-1] / total)
    return {"effective_rank": effective_rank, "top1_ratio": top1_ratio}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Embedding collapse diagnostics for WavLeJEPA",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--voxceleb-root", type=Path, required=True)
    parser.add_argument("--num-files", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--max-duration", type=float, default=10.0)
    parser.add_argument("--frames-per-file", type=int, default=50)
    parser.add_argument(
        "--feature-source",
        type=str,
        default="topk",
        choices=["topk", "context"],
    )
    parser.add_argument(
        "--project",
        action="store_true",
        help="Apply projector to frame embeddings before analysis.",
    )
    parser.add_argument("--num-pairs", type=int, default=50000)
    parser.add_argument("--max-cov-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    root = args.voxceleb_root
    if root.name != "wav":
        root = root / "wav"

    all_files = sorted(root.rglob("*.wav"))
    if not all_files:
        raise FileNotFoundError(f"No wav files found under {root}")

    rng = random.Random(args.seed)
    if args.num_files >= len(all_files):
        sample_files = all_files
    else:
        sample_files = rng.sample(all_files, args.num_files)

    model = _load_model(args.checkpoint)
    frames, file_ids = _collect_samples(
        model=model,
        audio_paths=sample_files,
        sample_rate=args.sample_rate,
        max_duration=args.max_duration,
        batch_size=args.batch_size,
        frames_per_file=args.frames_per_file,
        feature_source=args.feature_source,
        project=args.project,
        seed=args.seed,
    )

    norms = np.linalg.norm(frames, axis=1)
    cosine_stats = _cosine_stats(frames, file_ids, args.num_pairs, args.seed)
    cov_stats = _covariance_stats(frames, args.max_cov_samples, args.seed)

    print("Samples:", frames.shape[0])
    print("Embedding dim:", frames.shape[1])
    print("Norm mean/std:", norms.mean(), norms.std())
    print("Cosine stats:", cosine_stats)
    print("Covariance stats:", cov_stats)


if __name__ == "__main__":
    main()
