#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat << 'EOF'
Usage:
  bash depth/prepare_depths.sh --train-depth-dir <DIR> --test-depth-dir <DIR> --images-dir <DIR> --output-root <DIR> [options]

Required:
  --train-depth-dir   C3VD train depth dir (contains *_depth.tiff)
  --test-depth-dir    C3VD test depth dir (contains *_depth.tiff)
  --images-dir        Prepared images dir for name alignment (e.g., <OUT_ROOT>/images)
  --output-root       Prepared dataset root where outputs will be written

Outputs (under --output-root):
  - depths/        inverse-depth uint16 PNGs (main output, used by depth-scale computation / training)
  - depths_linear/ linear depth uint16 PNGs (optional)

Options:
  --split <train|test|both>   (default: both)
  --sample-interval <N>       (default: 10)
  --use-roi                   Enable ROI cropping
  --alpha <float>             (default: 0.0)
  --interp <linear|nearest>   (default: linear)
  --mask-thresh <float>       (default: 0.5)
  --z-min <float>             (default: 0.5)
  --z-max <float>             (default: 100.0)
  --dry-run                   Preview without writing

Env vars:
  TOOLS_DIR=<path/to/depth>  (default: this script's directory)
EOF
}

TRAIN_DEPTH_DIR=""
TEST_DEPTH_DIR=""
IMAGES_DIR=""
OUTPUT_ROOT=""

SPLIT="both"
SAMPLE_INTERVAL="10"
ALPHA="0.0"
INTERP="linear"
MASK_THRESH="0.5"
Z_MIN="0.5"
Z_MAX="100.0"
USE_ROI=""
DRY_RUN=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --train-depth-dir) TRAIN_DEPTH_DIR="$2"; shift 2;;
    --test-depth-dir) TEST_DEPTH_DIR="$2"; shift 2;;
    --images-dir) IMAGES_DIR="$2"; shift 2;;
    --output-root) OUTPUT_ROOT="$2"; shift 2;;
    --split) SPLIT="$2"; shift 2;;
    --sample-interval) SAMPLE_INTERVAL="$2"; shift 2;;
    --alpha) ALPHA="$2"; shift 2;;
    --interp) INTERP="$2"; shift 2;;
    --mask-thresh) MASK_THRESH="$2"; shift 2;;
    --z-min) Z_MIN="$2"; shift 2;;
    --z-max) Z_MAX="$2"; shift 2;;
    --use-roi) USE_ROI="--use-roi"; shift 1;;
    --dry-run) DRY_RUN="--dry-run"; shift 1;;
    -h|--help) usage; exit 0;;
    *) echo "[ERROR] Unknown argument: $1" >&2; usage; exit 2;;
  esac
done

if [[ -z "${TRAIN_DEPTH_DIR}" || -z "${TEST_DEPTH_DIR}" || -z "${IMAGES_DIR}" || -z "${OUTPUT_ROOT}" ]]; then
  echo "[ERROR] Missing required args." >&2
  usage
  exit 2
fi

TOOLS_DIR="${TOOLS_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

OUT_LINEAR_DIR="${OUTPUT_ROOT}/depths_linear"
OUT_INV_DIR="${OUTPUT_ROOT}/depths"

python "${TOOLS_DIR}/convert_c3vd_depth.py" \
  --split "${SPLIT}" \
  --train-depth-dir "${TRAIN_DEPTH_DIR}" \
  --test-depth-dir "${TEST_DEPTH_DIR}" \
  --images-dir "${IMAGES_DIR}" \
  --output-linear-dir "${OUT_LINEAR_DIR}" \
  --output-inv-dir "${OUT_INV_DIR}" \
  --alpha "${ALPHA}" \
  --interp "${INTERP}" \
  ${USE_ROI} \
  --mask-thresh "${MASK_THRESH}" \
  --z-min "${Z_MIN}" \
  --z-max "${Z_MAX}" \
  --sample-interval "${SAMPLE_INTERVAL}" \
  ${DRY_RUN}

echo "Done."
