#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/prepare_rgb_images.sh \
    --train-root <C3VD_TRAIN_ROOT> \
    --test-root <C3VD_TEST_ROOT> \
    --out-root <CURATED_SCENE_ROOT> [options]

Required:
  --train-root       C3VD training sequence root, containing rgb/
  --test-root        C3VD testing sequence root, containing rgb/
  --out-root         Output curated-style scene root

Options:
  --sample-interval  Sample test frames every N frames (default: 10)
  --clean-temp <0|1> Remove temporary undistorted folders (default: 1)

Outputs:
  <out-root>/images
  <out-root>/images_train
  <out-root>/images_test
  <out-root>/images_2
  <out-root>/images_4
  <out-root>/images_8
EOF
}

TRAIN_ROOT=""
TEST_ROOT=""
OUT_ROOT=""
SAMPLE_INTERVAL="10"
CLEAN_TEMP="1"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --train-root) TRAIN_ROOT="$2"; shift 2 ;;
    --test-root) TEST_ROOT="$2"; shift 2 ;;
    --out-root) OUT_ROOT="$2"; shift 2 ;;
    --sample-interval) SAMPLE_INTERVAL="$2"; shift 2 ;;
    --clean-temp) CLEAN_TEMP="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "${TRAIN_ROOT}" || -z "${TEST_ROOT}" || -z "${OUT_ROOT}" ]]; then
  echo "[ERROR] Missing required args." >&2
  usage
  exit 2
fi

if [[ ! -d "${TRAIN_ROOT}/rgb" ]]; then
  echo "[ERROR] train-root must contain rgb/: ${TRAIN_ROOT}" >&2
  exit 1
fi
if [[ ! -d "${TEST_ROOT}/rgb" ]]; then
  echo "[ERROR] test-root must contain rgb/: ${TEST_ROOT}" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "${OUT_ROOT}"

python "${SCRIPT_DIR}/undistort_rgb.py" \
  --input_v1 "${TRAIN_ROOT}/rgb" \
  --input_v2 "${TEST_ROOT}/rgb" \
  --output_v1 "${OUT_ROOT}/temp_undistorted_train" \
  --output_v2 "${OUT_ROOT}/temp_undistorted_test" \
  --camera_params_output "${OUT_ROOT}/temp_camera_params.txt"

rm -rf "${OUT_ROOT}/images" "${OUT_ROOT}/images_train" "${OUT_ROOT}/images_test"
mkdir -p "${OUT_ROOT}/images" "${OUT_ROOT}/images_train" "${OUT_ROOT}/images_test"

cp -f "${OUT_ROOT}/temp_undistorted_train"/*.png "${OUT_ROOT}/images/"
cp -f "${OUT_ROOT}/temp_undistorted_train"/*.png "${OUT_ROOT}/images_train/"

mapfile -t test_images < <(find "${OUT_ROOT}/temp_undistorted_test" -maxdepth 1 -type f -name '*.png' | sort)
if [[ ${#test_images[@]} -eq 0 ]]; then
  echo "[ERROR] No undistorted test images found." >&2
  exit 1
fi

new_idx=0
for ((orig_idx=0; orig_idx<${#test_images[@]}; orig_idx+=SAMPLE_INTERVAL)); do
  new_name="$(printf 'frame_1_%04d.png' "${new_idx}")"
  cp -f "${test_images[$orig_idx]}" "${OUT_ROOT}/images_test/${new_name}"
  cp -f "${test_images[$orig_idx]}" "${OUT_ROOT}/images/${new_name}"
  new_idx=$((new_idx + 1))
done

python "${SCRIPT_DIR}/create_multires_images.py" \
  --input_dir "${OUT_ROOT}/images" \
  --output_2 "${OUT_ROOT}/images_2" \
  --output_4 "${OUT_ROOT}/images_4" \
  --output_8 "${OUT_ROOT}/images_8"

if [[ "${CLEAN_TEMP}" == "1" ]]; then
  rm -rf "${OUT_ROOT}/temp_undistorted_train" "${OUT_ROOT}/temp_undistorted_test"
  rm -f "${OUT_ROOT}/temp_camera_params.txt"
fi

echo "[DONE] RGB images prepared: ${OUT_ROOT}"
