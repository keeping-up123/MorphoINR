#!/usr/bin/env bash
set -euo pipefail

# Reproduce the Human_Brain_prefrontal_cortex_151507 aligned AE + FiLM-SIREN experiment.
# Run this script from any directory; all paths are resolved relative to the
# MorphoINR project root.

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CASE="${1:-Human_Brain_prefrontal_cortex_151507}"
IMAGE_FILE="${2:-${PROJECT_ROOT}/data/random/${CASE}/tissue_hires_image.png}"
if [[ "${IMAGE_FILE}" != /* ]]; then
  IMAGE_FILE="${PROJECT_ROOT}/${IMAGE_FILE}"
fi

if [[ "${CASE}" != "Human_Brain_prefrontal_cortex_151507" ]]; then
  echo "This release is instantiated for Human_Brain_prefrontal_cortex_151507 only (received: ${CASE})." >&2
  exit 2
fi

DATA_DIR="${PROJECT_ROOT}/data/random/${CASE}"
RAW_FILE="${DATA_DIR}/${CASE}.h5ad"
PHIKON_FILE="${DATA_DIR}/${CASE}_with_phikon.h5ad"
PROCESSED_FILE="${DATA_DIR}/${CASE}_aealign_with_embeddings_phikon_zlocal.h5ad"
AE_CONFIG="${PROJECT_ROOT}/Configs/experiments/${CASE}/aealign.yaml"
INR_CONFIG="${PROJECT_ROOT}/Configs/experiments/${CASE}/inr_aealign_film_zlocal_siren.yaml"
AE_LOG_DIR="${PROJECT_ROOT}/log/random/${CASE}/aealign"

test -f "${AE_CONFIG}" || { echo "Missing AE config: ${AE_CONFIG}" >&2; exit 1; }
test -f "${INR_CONFIG}" || { echo "Missing FiLM-INR config: ${INR_CONFIG}" >&2; exit 1; }
test -f "${RAW_FILE}" || {
  echo "Missing input AnnData: ${RAW_FILE}" >&2
  echo "Place data/random/Human_Brain_prefrontal_cortex_151507/Human_Brain_prefrontal_cortex_151507.h5ad here (or obtain the licensed equivalent)." >&2
  exit 1
}

cd "${PROJECT_ROOT}"
if [[ ! -f "${PHIKON_FILE}" ]]; then
  test -f "${IMAGE_FILE}" || {
    echo "Missing histology image: ${IMAGE_FILE}" >&2
    echo "Pass its path as the second argument or place it beside ${RAW_FILE}." >&2
    exit 1
  }
  echo "[1/4] Extracting Phikon-v2 features"
  python Scripts/preprocess/extract_phikon_features.py \
    --h5ad "${RAW_FILE}" \
    --image "${IMAGE_FILE}" \
    --pixel_coord_key spatial_pixel \
    --output "${PHIKON_FILE}"
else
  echo "[1/4] Reusing existing Phikon input: ${PHIKON_FILE}"
fi

echo "[2/4] Training aligned AE: ${CASE}"
python Code/train.py --mode embedder --conf "${AE_CONFIG}"

if [[ ! -f "${PROCESSED_FILE}" ]]; then
  # Fallback for older Lightning versions/configurations without embedded_output.
  EMBEDDED_FILE="$(find "${AE_LOG_DIR}" -type f -name embedded-all.h5ad -print | sort | tail -n 1)"
  test -n "${EMBEDDED_FILE}" || {
    echo "No stage-1 output found at ${PROCESSED_FILE} or under ${AE_LOG_DIR}" >&2
    exit 1
  }
  mkdir -p "${DATA_DIR}"
  cp "${EMBEDDED_FILE}" "${PROCESSED_FILE}"
fi
echo "[3/4] Prepared FiLM-INR input: ${PROCESSED_FILE}"

echo "[4/4] Training FiLM-SIREN: ${CASE}"
python Code/train.py --mode inr_film --conf "${INR_CONFIG}"
echo "Finished ${CASE}; outputs are under log/random/${CASE}/inr_aealign_film_zlocal_siren."
