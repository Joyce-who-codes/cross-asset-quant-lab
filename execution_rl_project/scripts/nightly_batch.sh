#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/joyce/projects/cross-asset-quant-lab/execution_rl_project"

cd "${PROJECT_ROOT}"
mkdir -p logs

echo "===== nightly batch start $(date) ====="

bash scripts/merge_batch.sh
bash scripts/train_buy_phase1_batch.sh
bash scripts/train_buy_phase2_batch.sh
bash scripts/train_sell_phase1_batch.sh
bash scripts/train_sell_phase2_batch.sh

echo "===== nightly batch end $(date) ====="