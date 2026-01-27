#!/usr/bin/env python3
"""Analyze invariance/total loss drift across W&B runs.

Usage examples:
  scripts/analyze_wandb_invariance.py \
    --runs fine-hill-33 fresh-water-35 \
    --fetch

  scripts/analyze_wandb_invariance.py \
    --parquet analysis/wandb_runs_fine_hill_33_vs_fresh_water_35.parquet
"""

from __future__ import annotations

import argparse
import os
from typing import Iterable, Optional

import numpy as np
import pandas as pd


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return float("nan")
    x = x[mask]
    y = y[mask]
    x = x - x.mean()
    return float(np.polyfit(x, y, 1)[0])


def _tail_window(n: int, frac: float = 0.2, min_points: int = 50) -> int:
    return min(n, max(min_points, int(frac * n)))


def _fetch_runs(entity: Optional[str], project: str, run_names: Iterable[str]) -> pd.DataFrame:
    import wandb  # deferred import

    api = wandb.Api()
    if entity is None:
        entity = api.default_entity

    filters = {
        "$or": [
            {"name": name} for name in run_names
        ]
        + [
            {"display_name": name} for name in run_names
        ]
    }

    runs = {r.name: r for r in api.runs(f"{entity}/{project}", filters=filters)}
    rows = []

    keys = [
        "loss/total",
        "loss/invariance",
        "loss/sigreg",
        "train/loss",
        "lr",
        "optimizer/lr",
        "_step",
    ]

    for name in run_names:
        run = runs.get(name)
        if run is None:
            print(f"{name}: not found")
            continue

        # Pull fine-resolution history. history(samples=...) works even when scan_history fails.
        hist = run.history(pandas=True, samples=100000)
        if hist is None or hist.empty:
            print(f"{name}: no history")
            continue

        # Attach identifiers and config highlights
        hist["run_name"] = name
        hist["run_id"] = run.id
        config = {k: v for k, v in run.config.items() if not k.startswith("_")}
        hist["sigreg_weight"] = config.get("loss", {}).get("sigreg_weight")
        hist["peak_lr"] = config.get("optimizer", {}).get("peak_lr")
        rows.append(hist)

    if not rows:
        raise SystemExit("No data collected from W&B")

    df = pd.concat(rows, ignore_index=True)
    if "lr" not in df.columns or df["lr"].isna().all():
        if "optimizer/lr" in df.columns:
            df["lr"] = df["optimizer/lr"]
    return df


def _analyze(df: pd.DataFrame, match_step: Optional[float]) -> dict[str, dict[str, float]]:
    for col in ["loss/total", "loss/invariance", "loss/sigreg", "train/loss", "lr", "_step"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    summary: dict[str, dict[str, float]] = {}

    for run_name, g in df.groupby("run_name"):
        g = g.dropna(subset=["_step"]).sort_values("_step")
        if g.empty:
            continue
        last = g.iloc[-1]
        n = len(g)
        window = _tail_window(n)
        tail = g.iloc[-window:]

        summary[run_name] = {
            "points": float(n),
            "last_step": float(last["_step"]),
            "last_loss_total": float(last.get("loss/total", np.nan)),
            "last_loss_inv": float(last.get("loss/invariance", np.nan)),
            "last_loss_sigreg": float(last.get("loss/sigreg", np.nan)),
            "last_lr": float(last.get("lr", np.nan)),
            "inv_slope_tail": _slope(tail["_step"].to_numpy(), tail.get("loss/invariance").to_numpy()),
            "total_slope_tail": _slope(tail["_step"].to_numpy(), tail.get("loss/total").to_numpy()),
            "tail_window": float(window),
        }

        if "sigreg_weight" in g.columns:
            summary[run_name]["sigreg_weight"] = float(g["sigreg_weight"].dropna().iloc[-1])
        if "peak_lr" in g.columns:
            summary[run_name]["peak_lr"] = float(g["peak_lr"].dropna().iloc[-1])

    if match_step is not None:
        for run_name, g in df.groupby("run_name"):
            g = g.dropna(subset=["_step"]).sort_values("_step")
            if g.empty:
                continue
            idx = (g["_step"] - match_step).abs().idxmin()
            row = g.loc[idx]
            summary[run_name]["matched_step"] = float(row.get("_step", np.nan))
            summary[run_name]["matched_inv"] = float(row.get("loss/invariance", np.nan))
            summary[run_name]["matched_total"] = float(row.get("loss/total", np.nan))

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze W&B invariance drift.")
    parser.add_argument("--runs", nargs="+", help="Run names to fetch.")
    parser.add_argument("--entity", default=None, help="W&B entity (optional).")
    parser.add_argument("--project", default="wavlejepa", help="W&B project.")
    parser.add_argument("--fetch", action="store_true", help="Fetch from W&B API.")
    parser.add_argument(
        "--parquet",
        default=None,
        help="Existing parquet to analyze (skips fetch unless --fetch is set).",
    )
    parser.add_argument(
        "--out-parquet",
        default=None,
        help="Where to write fetched history (if --fetch).",
    )
    parser.add_argument(
        "--match-step",
        type=float,
        default=None,
        help="Step value to compare across runs (nearest step).",
    )

    args = parser.parse_args()

    df = None
    if args.fetch:
        if not args.runs:
            raise SystemExit("--runs is required with --fetch")
        df = _fetch_runs(args.entity, args.project, args.runs)
        out_path = args.out_parquet or os.path.join(
            "analysis",
            f"wandb_runs_{'_'.join(args.runs)}.parquet",
        )
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df.to_parquet(out_path, index=False)
        print(f"Wrote: {out_path}")

    if df is None:
        if not args.parquet:
            raise SystemExit("Provide --parquet or use --fetch")
        df = pd.read_parquet(args.parquet)

    summary = _analyze(df, args.match_step)
    for run_name, stats in summary.items():
        print("\n===", run_name, "===")
        for key, value in stats.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
