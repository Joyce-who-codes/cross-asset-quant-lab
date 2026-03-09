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