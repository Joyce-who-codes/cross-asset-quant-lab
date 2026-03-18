# Execution RL Project

Minimal RL execution project on Binance order book data with HFTBacktest.

## Goal
Train a single-asset execution agent:
- input: target quantity to buy within a finite horizon
- output: order placement decisions on order book
- benchmark against simple execution baselines

## MVP setup
- asset: BTCUSDT
- task: buy execution
- horizon: 30s
- action space:
  - HOLD
  - PLACE_BID1
  - PLACE_BID2
  - MARKET_BUY_SMALL
  - CANCEL_ALL

## Directory
- `src/data`: data preparation
- `src/backtest`: HFTBacktest wrapper
- `src/features`: state features
- `src/env`: Gym environment
- `src/baselines`: rule-based baselines
- `src/agents`: PPO train/eval

## Batch training workflow

Use a symbol list in [configs/symbol_batch_top5.txt](configs/symbol_batch_top5.txt) and run the batch scripts in order:

- [scripts/merge_batch.sh](scripts/merge_batch.sh)
- [scripts/train_buy_phase1_batch.sh](scripts/train_buy_phase1_batch.sh)
- [scripts/train_buy_phase2_batch.sh](scripts/train_buy_phase2_batch.sh)
- [scripts/train_sell_phase1_batch.sh](scripts/train_sell_phase1_batch.sh)
- [scripts/train_sell_phase2_batch.sh](scripts/train_sell_phase2_batch.sh)

The training entrypoint is [src/agents/ppo_train.py](src/agents/ppo_train.py) and now supports:

- `--symbol`
- `--side`
- `--phase`

Example nightly cron with lock and log:

`0 21 * * * cd /home/joyce/projects/cross-asset-quant-lab/execution_rl_project && flock -n /tmp/exec_rl_batch.lock bash scripts/nightly_batch.sh >> logs/nightly_batch.log 2>&1`