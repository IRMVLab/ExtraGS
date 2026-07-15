#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# OpenAI / proxy settings for scene caption generation.
#
# Fill these values if you want the pipeline to generate
# batch_captions_result_<scene>.jsonl automatically. This also
# supports OpenAI-compatible relay/proxy endpoints.
#
# Examples:
#   OPENAI_API_KEY="sk-..."
#   OPENAI_BASE_URL="https://your-relay.example.com/v1"
#   OPENAI_MODEL="gpt-4o"
#
# Leave OPENAI_API_KEY empty and run with --skip-gpt 1 if you
# prefer to generate captions later.
# ------------------------------------------------------------
OPENAI_API_KEY=""
OPENAI_BASE_URL=""
OPENAI_MODEL="gpt-4o"

usage() {
  cat << 'EOF'
Usage:
  bash prepare_c3vd_curated.sh --train-root <C3VD_TRAIN_ROOT> --test-root <C3VD_TEST_ROOT> --out-root <OUT_ROOT> [options]

Required:
  --train-root     C3VD train root (contains rgb/ depth/ pose.txt)
  --test-root      C3VD test root  (contains rgb/ depth/ pose.txt)
  --out-root       Output dataset root (curated-style structure will be created here)

Options:
  --sample-interval <N>    Test sampling interval (default: 10)
  --clean-temp <0|1>       Remove temp_* after run_all (default: 0)
  --colmap-bin <CMD>       COLMAP binary name/path (default: colmap)
  --cpu-threads <N>        Threads for COLMAP (default: 16)
  --matcher <NAME>         exhaustive_matcher|sequential_matcher... (default: exhaustive_matcher)
  --skip-gpt <0|1>         Skip GPT caption step (default: 0)

Notes:
  - Depth main output is <OUT_ROOT>/depths (inverse-depth uint16 PNGs)
  - Edit OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL near the top of this script for caption generation.
EOF
}

TRAIN_ROOT=""
TEST_ROOT=""
OUT_ROOT=""

SAMPLE_INTERVAL="10"
CLEAN_TEMP="1"
COLMAP_BIN="colmap"
CPU_THREADS="16"
MATCHER="exhaustive_matcher"
SKIP_GPT="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --train-root) TRAIN_ROOT="$2"; shift 2;;
    --test-root) TEST_ROOT="$2"; shift 2;;
    --out-root) OUT_ROOT="$2"; shift 2;;
    --sample-interval) SAMPLE_INTERVAL="$2"; shift 2;;
    --clean-temp) CLEAN_TEMP="$2"; shift 2;;
    --colmap-bin) COLMAP_BIN="$2"; shift 2;;
    --cpu-threads) CPU_THREADS="$2"; shift 2;;
    --matcher) MATCHER="$2"; shift 2;;
    --skip-gpt) SKIP_GPT="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "[ERROR] Unknown arg: $1" >&2; usage; exit 2;;
  esac
done

if [[ -z "${TRAIN_ROOT}" || -z "${TEST_ROOT}" || -z "${OUT_ROOT}" ]]; then
  echo "[ERROR] Missing required args." >&2
  usage
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RGB_PREP_SH="${SCRIPT_DIR}/scripts/prepare_rgb_images.sh"
DEPTH_PREP_SH="${SCRIPT_DIR}/depth/prepare_depths.sh"
COLMAP_SPLIT_SH="${SCRIPT_DIR}/colmap/build_and_split_colmap.sh"
DEPTH_SCALE_PY="${SCRIPT_DIR}/compute_depth_scale.py"
CAPTION_PY="${SCRIPT_DIR}/caption_scene.py"

mkdir -p "${OUT_ROOT}"
LOG_FILE="${OUT_ROOT}/pipeline.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "============================================================"
echo "C3VD -> curated pipeline"
echo "============================================================"
echo "[INFO] TRAIN_ROOT=${TRAIN_ROOT}"
echo "[INFO] TEST_ROOT =${TEST_ROOT}"
echo "[INFO] OUT_ROOT  =${OUT_ROOT}"
echo "[INFO] SAMPLE_INTERVAL=${SAMPLE_INTERVAL}  CLEAN_TEMP=${CLEAN_TEMP}"
echo "[INFO] COLMAP_BIN=${COLMAP_BIN}  CPU_THREADS=${CPU_THREADS}  MATCHER=${MATCHER}"
echo "[INFO] OPENAI_BASE_URL=${OPENAI_BASE_URL:-<official OpenAI endpoint>}"
echo "[INFO] OPENAI_MODEL=${OPENAI_MODEL}"
echo "[INFO] Log: ${LOG_FILE}"
echo "============================================================"

echo
echo "[STEP 1/5] prepare RGB images"
bash "${RGB_PREP_SH}" \
  --train-root "${TRAIN_ROOT}" \
  --test-root "${TEST_ROOT}" \
  --out-root "${OUT_ROOT}" \
  --sample-interval "${SAMPLE_INTERVAL}" \
  --clean-temp "${CLEAN_TEMP}"

echo
echo "[STEP 2/5] prepare depth maps"
bash "${DEPTH_PREP_SH}" \
  --train-depth-dir "${TRAIN_ROOT}/depth" \
  --test-depth-dir "${TEST_ROOT}/depth" \
  --images-dir "${OUT_ROOT}/images" \
  --output-root "${OUT_ROOT}" \
  --split both \
  --sample-interval "${SAMPLE_INTERVAL}"

echo
echo "[STEP 3/5] build and split COLMAP model"
bash "${COLMAP_SPLIT_SH}" \
  --out-root "${OUT_ROOT}" \
  --src-images "${OUT_ROOT}/images" \
  --colmap-bin "${COLMAP_BIN}" \
  --cpu-threads "${CPU_THREADS}" \
  --matcher "${MATCHER}" \
  --test-keyword "frame"

echo
echo "[STEP 4/5] compute depth scale"
python "${DEPTH_SCALE_PY}" \
  --base_dir "${OUT_ROOT}" \
  --depths_dir "${OUT_ROOT}/depths" \
  --model_type bin \
  --dataset nerfbusters-dataset

echo
echo "[STEP 5/5] caption scene"
if [[ "${SKIP_GPT}" == "1" ]]; then
  echo "[SKIP] --skip-gpt=1"
else
  if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "[ERROR] OPENAI_API_KEY is not set, but GPT step is enabled." >&2
    echo "        Edit OPENAI_API_KEY near the top of this script, or re-run with --skip-gpt 1" >&2
    exit 1
  fi
  SCENE_NAME="$(basename "$OUT_ROOT")"
  export OPENAI_API_KEY OPENAI_BASE_URL OPENAI_MODEL
  python "${CAPTION_PY}" \
    --source_path "${OUT_ROOT}" \
    --output "${OUT_ROOT}/batch_captions_result_${SCENE_NAME}.jsonl"
fi

echo
echo "=============================="
echo "[DONE] Pipeline finished"
echo "  OUT_ROOT=${OUT_ROOT}"
echo "  Log=${LOG_FILE}"
echo "=============================="
