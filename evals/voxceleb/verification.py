"""VoxCeleb verification utilities (trials, scoring, EER)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class Trial:
    """Single verification trial."""

    label: int
    enroll_path: Path
    test_path: Path


def _resolve_path(root: Path, rel_path: str) -> Path:
    """Resolve a trial path against the dataset root."""
    candidate = root / rel_path
    if candidate.exists():
        return candidate
    wav_root = root / "wav"
    candidate = wav_root / rel_path
    if candidate.exists():
        return candidate
    return candidate


def load_trials(trials_path: Path, voxceleb_root: Path) -> list[Trial]:
    """Load VoxCeleb verification trials.

    Expected format per line: "<label> <enroll_relpath> <test_relpath>".
    """
    trials: list[Trial] = []
    missing: list[Path] = []

    with trials_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 3:
                raise ValueError(f"Malformed trial line: {line}")
            label_str, enroll_rel, test_rel = parts
            label = int(label_str)
            enroll_path = _resolve_path(voxceleb_root, enroll_rel)
            test_path = _resolve_path(voxceleb_root, test_rel)
            if not enroll_path.exists():
                missing.append(enroll_path)
            if not test_path.exists():
                missing.append(test_path)
            trials.append(Trial(label=label, enroll_path=enroll_path, test_path=test_path))

    if missing:
        sample = "\n".join(str(p) for p in missing[:10])
        raise FileNotFoundError(
            f"Missing {len(missing)} trial files under {voxceleb_root}. "
            f"Sample missing files:\n{sample}"
        )

    return trials


def unique_trial_files(trials: Iterable[Trial]) -> list[Path]:
    """Return unique file paths from trials (stable order)."""
    seen = set()
    unique: list[Path] = []
    for trial in trials:
        for path in (trial.enroll_path, trial.test_path):
            if path not in seen:
                seen.add(path)
                unique.append(path)
    return unique


def l2_normalize(embeddings: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize embeddings along the last dimension."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, eps)


def cosine_scores(
    embeddings: np.ndarray,
    file_index: dict[Path, int],
    trials: Iterable[Trial],
) -> np.ndarray:
    """Compute cosine similarity scores for trials."""
    enroll_idx = []
    test_idx = []
    for trial in trials:
        enroll_idx.append(file_index[trial.enroll_path])
        test_idx.append(file_index[trial.test_path])
    enroll_idx = np.asarray(enroll_idx, dtype=np.int64)
    test_idx = np.asarray(test_idx, dtype=np.int64)

    enroll_emb = embeddings[enroll_idx]
    test_emb = embeddings[test_idx]
    return np.sum(enroll_emb * test_emb, axis=1)


def compute_eer(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Compute EER and the corresponding threshold.

    Returns:
        (eer, threshold)
    """
    labels = np.asarray(labels).astype(np.int32)
    scores = np.asarray(scores).astype(np.float64)

    if labels.shape[0] != scores.shape[0]:
        raise ValueError("Labels and scores must have the same length.")

    # Sort by descending score
    order = np.argsort(scores)[::-1]
    scores = scores[order]
    labels = labels[order]

    positives = np.sum(labels == 1)
    negatives = np.sum(labels == 0)
    if positives == 0 or negatives == 0:
        raise ValueError("Both positive and negative trials are required for EER.")

    tp = np.cumsum(labels == 1)
    fp = np.cumsum(labels == 0)
    frr = 1.0 - tp / positives
    far = fp / negatives

    diff = far - frr
    idx = np.argmin(np.abs(diff))
    eer = (far[idx] + frr[idx]) / 2.0
    threshold = scores[idx]
    return float(eer), float(threshold)
