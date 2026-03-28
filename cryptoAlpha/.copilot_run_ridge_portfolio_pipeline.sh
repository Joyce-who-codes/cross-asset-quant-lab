#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/joyce/projects/cross-asset-quant-lab/cryptoAlpha"
PYTHON_BIN="/home/joyce/.venv/bin/python"
LOG_DIR="$REPO_ROOT/.copilot_run_logs/ridge_portfolio"
STATUS_FILE="$LOG_DIR/status.txt"

mkdir -p "$LOG_DIR"

cd "$REPO_ROOT"

log_step() {
  local message="$1"
  printf '%s | %s\n' "$(date -Iseconds)" "$message" | tee -a "$STATUS_FILE"
}

: > "$STATUS_FILE"
log_step "START ridge portfolio pipeline"
log_step "REPO_ROOT=$REPO_ROOT"
log_step "PYTHON_BIN=$PYTHON_BIN"

log_step "STEP 1 run Ridge alpha train/save/backtest pipeline"
"$PYTHON_BIN" -u src/analysis/portfolio_ridge_pipeline.py \
  --horizon 1 \
  --rebalance-every-hours 1 \
  --portfolio-forward-hours 1 \
  --prediction-start 2025-12-18 \
  --prediction-end "2025-12-28 23:59:59" \
  --execution-start 2025-12-18 \
  --execution-end "2025-12-28 23:59:59" \
  > "$LOG_DIR/pipeline.log" 2>&1
log_step "STEP 1 done"

log_step "FINISH ridge portfolio pipeline"