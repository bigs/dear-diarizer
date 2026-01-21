"""Inspect a VoxCeleb verification trial list for sanity."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path


def _speaker_id_from_path(rel_path: str) -> str:
    parts = rel_path.split("/")
    return parts[1] if parts and parts[0] == "wav" and len(parts) > 1 else parts[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect VoxCeleb verification trial list",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument("--show-sample", type=int, default=5)
    args = parser.parse_args()

    labels = Counter()
    files = set()
    speakers = set()
    malformed = 0
    sample_lines = []

    with args.trials.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 3:
                malformed += 1
                continue
            label, enroll_rel, test_rel = parts
            labels[label] += 1
            files.add(enroll_rel)
            files.add(test_rel)
            speakers.add(_speaker_id_from_path(enroll_rel))
            speakers.add(_speaker_id_from_path(test_rel))
            if len(sample_lines) < args.show_sample:
                sample_lines.append(line)

    total = sum(labels.values())
    print(f"Trials: {total}")
    print(f"Malformed lines: {malformed}")
    print(f"Label counts: {dict(labels)}")
    print(f"Unique files: {len(files)}")
    print(f"Unique speakers (by path prefix): {len(speakers)}")
    if sample_lines:
        print("\nSample lines:")
        for line in sample_lines:
            print(line)


if __name__ == "__main__":
    main()
