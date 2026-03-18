#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/joyce/projects/cross-asset-quant-lab/execution_rl_project"
PYTHON_BIN="/home/joyce/.venv/bin/python"
SYMBOL_FILE="${PROJECT_ROOT}/configs/symbol_batch_top5.txt"
LOG_DIR="${PROJECT_ROOT}/logs"

mkdir -p "${LOG_DIR}"
cd "${PROJECT_ROOT}"

while IFS= read -r symbol; do
  [[ -z "${symbol}" ]] && continue
  [[ "${symbol}" =~ ^# ]] && continue

  echo "=== MERGE ${symbol} ==="
  LOG_FILE="${LOG_DIR}/merge_${symbol}.log"

  ${PYTHON_BIN} scripts/merge_local_tardis_symbol.py --symbol "${symbol}" >> "${LOG_FILE}" 2>&1
done < "${SYMBOL_FILE}"