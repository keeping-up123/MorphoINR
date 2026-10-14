#!/usr/bin/env bash
set -euo pipefail

# Reproduce the two-stage aligned AE + FiLM-SIREN pipeline for one case.
# Run this script from any directory; all paths are resolved relative to the
# MorphoINR project root.

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CASE="${1:-1142243F}"

case "${CASE}" in
  1142243F)
    INR_CONFIG="${PROJECT_ROOT}/Configs/experiments/${CASE}/08_inr_aealign_film_zlocal_siren.yaml"
    INR_STAGE="08_inr_aealign_film_zlocal_siren"
    ;;
  *)
    INR_CONFIG="${PROJECT_ROOT}/Configs/experiments/${CASE}/05_inr_aealign_film_zlocal_siren.yaml"
    INR_STAGE="05_inr_aealign_film_zlocal_siren"
    ;;
esac

AE_CONFIG="${PROJECT_ROOT}/Configs/experiments/${CASE}/02_aealign.yaml"
AE_LOG_DIR="${PROJECT_ROOT}/logs_final/random/${CASE}/02_aealign"
PROCESSED_DIR="${PROJECT_ROOT}/data_final/random/${CASE}"
PROCESSED_FILE="${PROCESSED_DIR}/${CASE}_aealign_with_embeddings_phikon_zlocal.h5ad"

test -f "${AE_CONFIG}" || { echo "Missing AE config: ${AE_CONFIG}" >&2; exit 1; }
test -f "${INR_CONFIG}" || { echo "Missing FiLM-INR config: ${INR_CONFIG}" >&2; exit 1; }

cd "${PROJECT_ROOT}"
echo "[1/3] Training aligned AE: ${CASE}"
python Code/train.py --mode embedder --conf "${AE_CONFIG}"

if [[ ! -f "${PROCESSED_FILE}" ]]; then
  # Backward-compatible fallback for configurations without embedded_output.
  EMBEDDED_FILE="$(find "${AE_LOG_DIR}" -type f -name embedded-all.h5ad -print | sort | tail -n 1)"
  test -n "${EMBEDDED_FILE}" || {
    echo "No stage-1 output found at ${PROCESSED_FILE} or under ${AE_LOG_DIR}" >&2
    exit 1
  }
  mkdir -p "${PROCESSED_DIR}"
  cp "${EMBEDDED_FILE}" "${PROCESSED_FILE}"
fi
echo "[2/3] Prepared FiLM-INR input: ${PROCESSED_FILE}"

echo "[3/3] Training FiLM-SIREN: ${CASE}"
python Code/train.py --mode inr_film --conf "${INR_CONFIG}"
echo "Finished ${CASE}; outputs are under logs_final/random/${CASE}/${INR_STAGE}."
