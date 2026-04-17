#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 <config.yaml> <checkpoint.ckpt> <predict_input_dir> <predict_output_dir> [platform]"
  exit 1
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="$1"
CKPT="$2"
INPUT_DIR="$3"
OUTPUT_DIR="$4"
PLATFORM="${5:-landsat-c2-l2}"

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

echo "[predict] ${TT_CMD[*]} predict"
"${TT_CMD[@]}" predict \
  --config "${CONFIG}" \
  --ckpt_path "${CKPT}" \
  --predict_output_dir "${OUTPUT_DIR}" \
  --data.init_args.predict_data_root "${INPUT_DIR}"
