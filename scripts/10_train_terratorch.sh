#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <config.yaml> [platform]"
  exit 1
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="$1"
PLATFORM="${2:-landsat-c2-l2}"

if [[ "$CONFIG" != /* ]]; then
  CONFIG="$PWD/$CONFIG"
fi

TT_CMD=(terratorch)
if ! command -v terratorch >/dev/null 2>&1; then
  TT_CMD=(python -m terratorch)
fi

echo "[sync] updating means/stds from $ROOT/configs/metadata.yaml platform=$PLATFORM"
python "$ROOT/scripts/04_compute_stats.py" \
  --metadata-yaml "$ROOT/configs/metadata.yaml" \
  --platform "$PLATFORM" \
  --config "$CONFIG"

echo "[train] ${TT_CMD[*]} fit --config ${CONFIG}"
"${TT_CMD[@]}" fit --config "${CONFIG}"
