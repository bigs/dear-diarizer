"""
Manual checkpoint tarball upload utility.

This matches the naming + tarball contents used by the training uploader in
`wavlejepa.training.checkpoint_upload`.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from wavlejepa.training.checkpoint_upload import upload_checkpoint_tarball


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Tar and upload a local Orbax checkpoint directory to object storage "
            "using the same format as training's checkpoint uploader."
        )
    )
    parser.add_argument(
        "checkpoint",
        type=Path,
        help=(
            "Checkpoint path. Examples: /run/checkpoints/123, /run/best/123, "
            "or /run (use --step/--latest)."
        ),
    )
    parser.add_argument(
        "--bucket-uri",
        default=os.getenv("CHECKPOINT_BUCKET_PATH"),
        help=(
            "Bucket URI like s3://bucket/prefix or gs://bucket/prefix "
            "(defaults to env CHECKPOINT_BUCKET_PATH)."
        ),
    )
    parser.add_argument(
        "--step", type=int, default=None, help="Checkpoint step to upload."
    )
    parser.add_argument(
        "--best",
        dest="is_best",
        action="store_true",
        help="Upload from best/<step> and update remote best pointer.",
    )
    parser.add_argument(
        "--no-best",
        dest="is_best",
        action="store_false",
        help="Upload from checkpoints/<step> (default when not inferred from path).",
    )
    parser.set_defaults(is_best=None)
    parser.add_argument(
        "--latest",
        action="store_true",
        help="If --step is not set, pick the latest numeric step in the chosen subdir.",
    )
    parser.add_argument(
        "--no-wait-stable",
        action="store_true",
        help="Do not wait for the checkpoint directory to stabilize before tarring.",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=300.0,
        help="Max seconds to wait for checkpoint stabilization (default: 300).",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    if not args.bucket_uri:
        raise SystemExit(
            "Missing bucket target. Provide --bucket-uri or set CHECKPOINT_BUCKET_PATH."
        )

    tarball_name = upload_checkpoint_tarball(
        args.checkpoint,
        bucket_uri=args.bucket_uri,
        step=args.step,
        is_best=args.is_best,
        latest=args.latest,
        wait_for_stable=not args.no_wait_stable,
        timeout_s=args.timeout_s,
    )
    print(tarball_name)


if __name__ == "__main__":
    main()
