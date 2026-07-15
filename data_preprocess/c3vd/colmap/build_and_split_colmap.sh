#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat << 'EOF'
Usage:
  bash colmap/build_and_split_colmap.sh --out-root <OUT_ROOT> [options]

Required:
  --out-root <DIR>         Curated dataset root (contains images/)

Options:
  --src-images <DIR>       Source images dir (default: <OUT_ROOT>/images)
  --link-mode <copy|symlink>  How to stage images for COLMAP (default: symlink)
  --colmap-bin <CMD>       COLMAP binary (default: colmap)
  --cpu-threads <N>        CPU threads for SIFT/matching/mapper (default: 16)
  --matcher <NAME>         exhaustive|sequential|vocab_tree|spatial (also accepts *_matcher) (default: exhaustive)
  --camera-model <NAME>    COLMAP camera model (default: OPENCV)
  --single-camera <0|1>    Use single camera (default: 1)
  --seq-overlap <N>        Overlap for sequential matcher (default: 10)
  --test-keyword <KW>      Keyword to split test images (default: frame)
EOF
}

# =========================================================
# Args
# =========================================================
OUT_ROOT=""
SRC_IMAGES_DIR=""
LINK_MODE="symlink"

COLMAP_BIN="colmap"
CPU_THREADS="16"

COLMAP_CAMERA_MODEL="OPENCV"
COLMAP_SINGLE_CAMERA=1
COLMAP_MATCHER="exhaustive"
SEQ_OVERLAP=10
TEST_KEYWORD="frame"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out-root) OUT_ROOT="$2"; shift 2;;
    --src-images) SRC_IMAGES_DIR="$2"; shift 2;;
    --link-mode) LINK_MODE="$2"; shift 2;;
    --colmap-bin) COLMAP_BIN="$2"; shift 2;;
    --cpu-threads) CPU_THREADS="$2"; shift 2;;
    --matcher) COLMAP_MATCHER="$2"; shift 2;;
    --camera-model) COLMAP_CAMERA_MODEL="$2"; shift 2;;
    --single-camera) COLMAP_SINGLE_CAMERA="$2"; shift 2;;
    --seq-overlap) SEQ_OVERLAP="$2"; shift 2;;
    --test-keyword) TEST_KEYWORD="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "[ERROR] Unknown arg: $1" >&2; usage; exit 2;;
  esac
done

if [[ -z "${OUT_ROOT}" ]]; then
  echo "[ERROR] --out-root is required" >&2
  usage
  exit 2
fi

OUT_ROOT="$(realpath "${OUT_ROOT}")"
if [[ -z "${SRC_IMAGES_DIR}" ]]; then
  SRC_IMAGES_DIR="${OUT_ROOT}/images"
else
  SRC_IMAGES_DIR="$(realpath "${SRC_IMAGES_DIR}")"
fi

# Accept both short matcher names and COLMAP command-style names.
case "${COLMAP_MATCHER}" in
  exhaustive_matcher) COLMAP_MATCHER="exhaustive";;
  sequential_matcher) COLMAP_MATCHER="sequential";;
  vocab_tree_matcher) COLMAP_MATCHER="vocab_tree";;
  spatial_matcher) COLMAP_MATCHER="spatial";;
esac

if ! command -v "${COLMAP_BIN}" >/dev/null 2>&1; then
  echo "[ERROR] COLMAP not found: ${COLMAP_BIN}" >&2
  exit 1
fi

# =========================================================
# 0) User config: where are the original images?
# =========================================================
# SRC_IMAGES_DIR / LINK_MODE / COLMAP_* configured above

# =========================================================
# CPU ONLY switches (important)
# =========================================================
# 这些会传给 feature_extractor / matcher，强制不用GPU
CPU_SIFT_EXTRACT_FLAGS=(--SiftExtraction.use_gpu 0 --SiftExtraction.gpu_index -1)
CPU_SIFT_MATCH_FLAGS=(--SiftMatching.use_gpu 0 --SiftMatching.gpu_index -1)

# =========================================================
# 1) Paths
# =========================================================
BASE="${OUT_ROOT}/colmap"
MODEL_TXT="${BASE}/model_txt"
MODEL_BIN="${BASE}/sparse/0"
DB="${BASE}/database.db"
IMAGE_PATH="${BASE}/images"
WORKDIR="${BASE}/subset_work"

DEPTH_PARAMS_SRC="${BASE}/depth_params.json"  # optional

# Start from a clean COLMAP workspace under OUT_ROOT
rm -rf "${BASE}"
mkdir -p "${BASE}" "${WORKDIR}" "${IMAGE_PATH}"

# =========================================================
# 2) Prepare ./images from SRC_IMAGES_DIR
# =========================================================
echo "=============================="
echo "[STEP] Prepare ./images"
echo "[SRC] ${SRC_IMAGES_DIR}"
echo "[DST] ${IMAGE_PATH}"
echo "[MODE] ${LINK_MODE}"
echo "=============================="

if [ ! -d "${SRC_IMAGES_DIR}" ]; then
  echo "[ERROR] SRC_IMAGES_DIR not found: ${SRC_IMAGES_DIR}"
  exit 1
fi

rm -rf "${IMAGE_PATH}"
mkdir -p "${IMAGE_PATH}"

shopt -s nullglob
imgs=("${SRC_IMAGES_DIR}"/*.png "${SRC_IMAGES_DIR}"/*.jpg "${SRC_IMAGES_DIR}"/*.jpeg)
shopt -u nullglob
if [ ${#imgs[@]} -eq 0 ]; then
  echo "[ERROR] No images found in ${SRC_IMAGES_DIR} (png/jpg/jpeg)."
  exit 1
fi

if [ "${LINK_MODE}" = "copy" ]; then
  cp -f "${SRC_IMAGES_DIR}"/*.{png,jpg,jpeg} "${IMAGE_PATH}" 2>/dev/null || true
elif [ "${LINK_MODE}" = "symlink" ]; then
  for f in "${SRC_IMAGES_DIR}"/*.{png,jpg,jpeg}; do
    [ -e "$f" ] || continue
    ln -s "$f" "${IMAGE_PATH}/$(basename "$f")"
  done
else
  echo "[ERROR] LINK_MODE must be 'copy' or 'symlink'. Got: ${LINK_MODE}"
  exit 1
fi

echo "[INFO] images prepared: $(ls -1 "${IMAGE_PATH}" | wc -l)"

# =========================================================
# 3) Run COLMAP on ALL images (CPU ONLY)
# =========================================================
echo "=============================="
echo "[STEP] Run COLMAP (ALL) - CPU ONLY"
echo "[OUT]  ${BASE}"
echo "=============================="

# 3.1 feature extraction (CPU SIFT)
"${COLMAP_BIN}" feature_extractor \
  --database_path "${DB}" \
  --image_path "${IMAGE_PATH}" \
  --ImageReader.camera_model "${COLMAP_CAMERA_MODEL}" \
  --ImageReader.single_camera "${COLMAP_SINGLE_CAMERA}" \
  --SiftExtraction.num_threads "${CPU_THREADS}" \
  "${CPU_SIFT_EXTRACT_FLAGS[@]}"

# 3.2 matching (CPU SIFT matching)
if [ "${COLMAP_MATCHER}" = "exhaustive" ]; then
  "${COLMAP_BIN}" exhaustive_matcher \
    --database_path "${DB}" \
    --SiftMatching.num_threads "${CPU_THREADS}" \
    "${CPU_SIFT_MATCH_FLAGS[@]}"
elif [ "${COLMAP_MATCHER}" = "sequential" ]; then
  "${COLMAP_BIN}" sequential_matcher \
    --database_path "${DB}" \
    --SequentialMatching.overlap "${SEQ_OVERLAP}" \
    --SiftMatching.num_threads "${CPU_THREADS}" \
    "${CPU_SIFT_MATCH_FLAGS[@]}"
elif [ "${COLMAP_MATCHER}" = "vocab_tree" ]; then
  "${COLMAP_BIN}" vocab_tree_matcher \
    --database_path "${DB}" \
    --SiftMatching.num_threads "${CPU_THREADS}" \
    "${CPU_SIFT_MATCH_FLAGS[@]}"
elif [ "${COLMAP_MATCHER}" = "spatial" ]; then
  "${COLMAP_BIN}" spatial_matcher \
    --database_path "${DB}" \
    --SiftMatching.num_threads "${CPU_THREADS}" \
    "${CPU_SIFT_MATCH_FLAGS[@]}"
else
  echo "[ERROR] Unknown COLMAP_MATCHER=${COLMAP_MATCHER}"
  exit 1
fi

# 3.3 mapping (CPU)
mkdir -p "${BASE}/sparse"
"${COLMAP_BIN}" mapper \
  --database_path "${DB}" \
  --image_path "${IMAGE_PATH}" \
  --output_path "${BASE}/sparse" \
  --Mapper.num_threads "${CPU_THREADS}"

# 3.4 export TXT (needed by subset code)
mkdir -p "${MODEL_TXT}"
"${COLMAP_BIN}" model_converter \
  --input_path "${MODEL_BIN}" \
  --output_path "${MODEL_TXT}" \
  --output_type TXT

echo "[DONE] COLMAP all-model created:"
echo "       - ${MODEL_BIN}"
echo "       - ${MODEL_TXT}"

# =========================================================
# 4) Subset logic
# =========================================================
ALL_NAMES="${WORKDIR}/all_image_names.txt"
awk 'NR%2==1{print $NF}' "${MODEL_TXT}/images.txt" > "${ALL_NAMES}"

TRAIN_NAMES="${WORKDIR}/train_names.txt"
TEST_NAMES="${WORKDIR}/test_names.txt"

grep -F "${TEST_KEYWORD}" "${ALL_NAMES}" > "${TEST_NAMES}"  || true
grep -Fv "${TEST_KEYWORD}" "${ALL_NAMES}" > "${TRAIN_NAMES}" || true

echo "[INFO] total images: $(wc -l < "${ALL_NAMES}")"
echo "[INFO] train images: $(wc -l < "${TRAIN_NAMES}")"
echo "[INFO] test images : $(wc -l < "${TEST_NAMES}")"

if [ ! -s "${TRAIN_NAMES}" ]; then
  echo "[ERROR] TRAIN_NAMES is empty. Check split rule: TEST_KEYWORD='${TEST_KEYWORD}'"
  exit 1
fi
if [ ! -s "${TEST_NAMES}" ]; then
  echo "[WARN] TEST_NAMES is empty. Maybe no filenames contain '${TEST_KEYWORD}'. Continue..."
fi

run_subset() {
  local tag="$1"
  local del_list="$2"

  local OUT_ROOT="${BASE}/../model_${tag}"
  local OUT_SPARSE="${OUT_ROOT}/sparse"

  local TMP_FIXED="${WORKDIR}/tmp_${tag}_fixedPose"
  local TMP_TRI="${WORKDIR}/tmp_${tag}_triangulated"

  echo "=============================="
  echo "[RUN] subset=${tag}"
  echo "[RUN] delete_list=${del_list}"
  echo "[OUT] ${OUT_SPARSE}"
  echo "=============================="

  rm -rf "${OUT_ROOT}" "${TMP_FIXED}" "${TMP_TRI}"
  mkdir -p "${OUT_SPARSE}" "${TMP_FIXED}" "${TMP_TRI}"

  "${COLMAP_BIN}" image_deleter \
    --input_path "${MODEL_BIN}" \
    --output_path "${TMP_FIXED}" \
    --image_names_path "${del_list}"

  "${COLMAP_BIN}" point_triangulator \
    --database_path "${DB}" \
    --image_path "${IMAGE_PATH}" \
    --input_path "${TMP_FIXED}" \
    --output_path "${TMP_TRI}"

  if ls "${TMP_TRI}"/*.bin >/dev/null 2>&1; then
    cp -f "${TMP_TRI}/"*.bin "${OUT_SPARSE}/"
  elif [ -d "${TMP_TRI}/0" ] && ls "${TMP_TRI}/0"/*.bin >/dev/null 2>&1; then
    cp -f "${TMP_TRI}/0/"*.bin "${OUT_SPARSE}/"
  else
    echo "[ERROR] Cannot find *.bin in triangulated output: ${TMP_TRI}"
    exit 1
  fi

  "${COLMAP_BIN}" model_converter \
    --input_path "${OUT_SPARSE}" \
    --output_path "${OUT_ROOT}" \
    --output_type TXT

  "${COLMAP_BIN}" model_converter \
    --input_path "${OUT_SPARSE}" \
    --output_path "${OUT_SPARSE}/points3D.ply" \
    --output_type PLY

  if [ "${tag}" = "train" ] && [ -f "${DEPTH_PARAMS_SRC}" ]; then
    cp -f "${DEPTH_PARAMS_SRC}" "${OUT_SPARSE}/depth_params.json"
    echo "[INFO] Copied depth params -> ${OUT_SPARSE}/depth_params.json"
  fi

  echo "[DONE] ${tag} -> ${OUT_ROOT}"
}

DEL_NON_TRAIN="${WORKDIR}/delete_non_train.txt"
cp -f "${TEST_NAMES}" "${DEL_NON_TRAIN}"
run_subset "train" "${DEL_NON_TRAIN}"

DEL_TRAIN="${WORKDIR}/delete_train.txt"
cp -f "${TRAIN_NAMES}" "${DEL_TRAIN}"

if [ -s "${TEST_NAMES}" ]; then
  run_subset "test" "${DEL_TRAIN}"
else
  echo "[SKIP] test subset is empty, skip generating test model."
fi

echo "=============================="
echo "[ALL DONE]"
echo "ALL model : ${MODEL_BIN}"
echo "TRAIN     : ${BASE}/../model_train"
echo "TEST      : ${BASE}/../model_test"
echo "=============================="
