#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  upload_shards.sh --source /path/to/shards --bucket my-bucket
                   [--prefix path/in/bucket] [--dry-run] [--delete-extra]
                   [--project PROJECT]

Examples:
  upload_shards.sh --source ./shards --bucket my-bucket --prefix wavlejepa/train
  upload_shards.sh --source ./shards --bucket my-bucket --dry-run
USAGE
}

SOURCE=""
BUCKET=""
PREFIX=""
DRY_RUN=0
DELETE_EXTRA=0
PROJECT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source) SOURCE="$2"; shift 2 ;;
    --bucket) BUCKET="$2"; shift 2 ;;
    --prefix) PREFIX="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --delete-extra) DELETE_EXTRA=1; shift ;;
    --project) PROJECT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$SOURCE" || -z "$BUCKET" ]]; then
  echo "Error: --source and --bucket are required." >&2
  usage
  exit 2
fi

if [[ ! -d "$SOURCE" ]]; then
  echo "Error: source directory not found: $SOURCE" >&2
  exit 2
fi

if ! command -v gsutil >/dev/null 2>&1; then
  echo "Error: gsutil not found. Install the Google Cloud SDK and run 'gcloud auth login'." >&2
  exit 2
fi

DEST="gs://${BUCKET}"
if [[ -n "$PREFIX" ]]; then
  PREFIX="${PREFIX#/}"
  PREFIX="${PREFIX%/}"
  DEST="${DEST}/${PREFIX}"
fi

ARGS=(-m rsync -r -c)
if [[ "$DRY_RUN" -eq 1 ]]; then
  ARGS+=(-n)
fi
if [[ "$DELETE_EXTRA" -eq 1 ]]; then
  ARGS+=(-d)
fi
if [[ -n "$PROJECT" ]]; then
  ARGS+=(-p "$PROJECT")
fi

echo "Syncing: $SOURCE -> $DEST"
gsutil "${ARGS[@]}" "$SOURCE" "$DEST"
