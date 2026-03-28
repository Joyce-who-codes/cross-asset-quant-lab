#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/joyce/projects/cross-asset-quant-lab/execution_rl_project"
PYTHON_BIN="/home/joyce/.venv/bin/python"
LOG_DIR="$REPO_ROOT/.copilot_run_logs/btcusdt_buy"
STATUS_FILE="$LOG_DIR/status.txt"
SELF_CHECK_FILE="$LOG_DIR/chunk_self_check.txt"

mkdir -p "$LOG_DIR"

export TARDIS_RAW_ROOT="/home/joyce/projects/data/raw/tardis"
export TARDIS_CHUNK_ROOT="/home/joyce/projects/data/raw/tardis_chunks"

cd "$REPO_ROOT"

log_step() {
  local message="$1"
  printf '%s | %s\n' "$(date -Iseconds)" "$message" | tee -a "$STATUS_FILE"
}

: > "$STATUS_FILE"
log_step "START pipeline"
log_step "TARDIS_RAW_ROOT=$TARDIS_RAW_ROOT"
log_step "TARDIS_CHUNK_ROOT=$TARDIS_CHUNK_ROOT"

log_step "STEP 2A build train chunks 2025-12-01..2025-12-20"
"$PYTHON_BIN" -u scripts/build_tardis_chunks.py \
  --symbol BTCUSDT \
  --start-day 2025-12-01 \
  --end-day 2025-12-20 \
  --chunk-hours 6 \
  > "$LOG_DIR/step2_train_chunks.log" 2>&1
log_step "STEP 2A done"

log_step "STEP 2B build test chunks 2025-12-26..2026-01-01"
"$PYTHON_BIN" -u scripts/build_tardis_chunks.py \
  --symbol BTCUSDT \
  --start-day 2025-12-26 \
  --end-day 2026-01-01 \
  --chunk-hours 6 \
  > "$LOG_DIR/step2_test_chunks.log" 2>&1
log_step "STEP 2B done"

log_step "STEP 3 train PPO both phases"
"$PYTHON_BIN" -u -m src.agents.ppo_train \
  --symbol BTCUSDT \
  --side buy \
  --phase both \
  --train-config configs/train_btc_long.yaml \
  > "$LOG_DIR/step3_train_both.log" 2>&1
log_step "STEP 3 done"

log_step "STEP 4 evaluate single test chunk index 0"
"$PYTHON_BIN" -u -m src.agents.evaluate \
  --symbol BTCUSDT \
  --side buy \
  --chunk-index 0 \
  --train-config configs/train_btc_long.yaml \
  > "$LOG_DIR/step4_evaluate_chunk0.log" 2>&1
log_step "STEP 4 done"

log_step "STEP 5 evaluate all test chunks"
"$PYTHON_BIN" -u -m src.agents.evaluate_many \
  --symbol BTCUSDT \
  --side buy \
  --train-config configs/train_btc_long.yaml \
  > "$LOG_DIR/step5_evaluate_many.log" 2>&1
log_step "STEP 5 done"

log_step "STEP 6 self-check chunk files"
find "$TARDIS_CHUNK_ROOT/BTCUSDT" -maxdepth 2 -name meta.json | sort | head > "$SELF_CHECK_FILE"
log_step "STEP 6 done"
log_step "FINISH pipeline"
