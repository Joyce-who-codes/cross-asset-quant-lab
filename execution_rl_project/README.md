# Execution RL Project

一个基于高频订单簿回放的加密资产强化学习执行框架。项目使用 Tardis Binance Futures 逐笔订单簿、成交和快照数据构建 Gymnasium 环境，并使用 Stable-Baselines3 PPO 学习在有限时间内完成买入或卖出目标。

当前实现支持多交易对、买卖双向、分阶段训练、批量任务，以及与 TWAP 和被动挂单策略进行同起点对比评估。

## 项目目标

给定资产、交易方向、目标数量和执行时限，智能体需要在成本与完成率之间取得平衡：

- 输入：订单簿状态、近期成交特征和任务完成进度
- 输出：保持、被动挂单、市价小单或撤单
- 目标：降低 implementation shortfall，同时避免到期未完成
- 对照：TWAP Market、Passive Best Bid/Ask、Passive Then Sweep

该项目研究的是订单执行，不负责生成买卖信号。上游目标仓位可以由相邻的 `cryptoAlpha` 项目产生。

## 核心功能

- Tardis 增量 L2、逐笔成交和 25 档快照解析
- CSV/CSV.GZ 到标准 Parquet 时间块的转换
- 自研事件驱动订单簿回放与成交模拟
- 支持买入与卖出任务的 Gymnasium 环境
- PPO 训练、断点续训和观测归一化
- 固定数据块预热、局部随机训练和全数据块训练
- 单数据块及多数据块样本外评估
- 多交易对、多方向批量训练脚本

## 项目结构

```text
execution_rl_project/
├── configs/                         # 环境、数据和训练配置
├── scripts/                         # 数据准备、训练和批处理脚本
├── src/
│   ├── data/                        # Tardis 数据解析与样本构建
│   ├── backtest/                    # 订单簿回放及执行包装器
│   ├── features/                    # LOB 与任务状态特征
│   ├── env/                         # Gymnasium 环境和奖励函数
│   ├── baselines/                   # TWAP 与被动执行基准
│   ├── agents/                      # PPO 训练、评估和结果分析
│   ├── microalpha/                  # 微观 Alpha 模型实验
│   └── utils/                       # 路径、随机种子和数据块工具
├── results/                         # 模型、归一化器和评估结果
├── logs/                            # 批量任务日志（运行后生成）
└── requirements.txt
```

## 环境安装

建议使用 Python 3.10 或以上版本，并创建独立虚拟环境：

```bash
cd execution_rl_project
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

主要依赖包括：

- `gymnasium`
- `stable-baselines3`
- `torch`
- `pandas`、`numpy`、`pyarrow`
- `hftbacktest`

训练速度取决于数据量、订单簿回放开销和计算设备。默认配置使用 CPU。

## 数据准备

项目不附带 Tardis 原始数据。每个交易对需要以下三类 Binance Futures 数据：

```text
incremental_book_L2/<SYMBOL>/<YYYY-MM-DD>/<SYMBOL>.csv[.gz]
trades/<SYMBOL>/<YYYY-MM-DD>/<SYMBOL>.csv[.gz]
book_snapshot_25/<SYMBOL>/<YYYY-MM-DD>/<SYMBOL>.csv[.gz]
```

### 1. 配置本地数据路径

编辑 [`configs/batch_data.yaml`](configs/batch_data.yaml)：

```yaml
local_root: /path/to/local/tardis

splits:
  train:
    start_day: 2025-12-01
    end_day: 2025-12-20
  val:
    start_day: 2025-12-21
    end_day: 2025-12-25
  test:
    start_day: 2025-12-26
    end_day: 2026-01-01
```

同时核对 [`configs/train_btc_long.yaml`](configs/train_btc_long.yaml) 中的训练、验证、测试日期是否与数据覆盖范围一致。

### 2. 构建标准时间块

推荐直接将逐日原始数据转换成训练使用的 Parquet 块：

```bash
python scripts/build_tardis_chunks.py \
  --symbol BTCUSDT \
  --start-day 2025-12-01 \
  --end-day 2026-01-01 \
  --chunk-hours 6 \
  --raw-root /path/to/raw/tardis \
  --chunk-root /path/to/tardis_chunks
```

输出结构为：

```text
tardis_chunks/
└── BTCUSDT/
    ├── 2025-12-01_00/
    │   ├── book.parquet
    │   ├── trades.parquet
    │   ├── snapshot.parquet
    │   └── meta.json
    └── 2025-12-01_06/
        └── ...
```

默认每块 6 小时，`chunk-hours` 必须是 24 的正因数。已有块默认跳过；明确需要覆盖时使用 `--force`。

`src/utils/tardis_chunk.py` 内含默认数据块根目录。若不希望依赖该默认值，请在训练和评估命令中始终传入 `--chunk-root`。

## 执行环境

### 动作空间

环境使用五个离散动作：

```text
0  HOLD
1  PLACE_PASSIVE_1
2  PLACE_PASSIVE_2
3  MARKET_SMALL
4  CANCEL_ALL
```

买入时，两个被动动作分别在最佳买价和最佳买价下一个 tick 挂单；卖出时则分别在最佳卖价和最佳卖价上一个 tick 挂单。`MARKET_SMALL` 每次最多执行 `market_clip_qty`。

### 状态空间

当前观测为 8 维连续向量：6 个订单簿/近期成交特征，加上 2 个执行进度特征。具体构造逻辑位于 `src/features/lob_features.py`。

### 奖励函数

环境支持两类奖励：

- `shortfall`：根据相对到达价的成交成本给出逐步奖励，并对到期剩余数量施加惩罚
- TWAP-relative：在 shortfall 基础上加入相对 TWAP 进度、taker 成交和超额成本项

相关权重可通过训练配置中的 `lambda_terminal_remain`、`lambda_lag`、`lambda_taker`、`lambda_excess` 和 `reward_mode` 调整。

## 快速开始

以下命令均从 `execution_rl_project` 根目录执行。

### 1. 训练单个 PPO 智能体

```bash
python -m src.agents.ppo_train \
  --symbol BTCUSDT \
  --side buy \
  --phase both \
  --split train \
  --train-config configs/train_btc_long.yaml \
  --chunk-root /path/to/tardis_chunks
```

关键参数：

- `--symbol`：交易对，例如 `BTCUSDT`
- `--side`：`buy` 或 `sell`
- `--phase`：`phase1`、`phase2a`、`phase2b`、`phase2` 或 `both`
- `--split`：使用配置中的 `train`、`val` 或 `test` 日期区间
- `--max-chunks`：只加载前 N 个块，适合快速调试
- `--chunk-root`：覆盖默认数据块目录

训练产物默认保存为：

```text
results/checkpoints_<symbol>_<side>_1m/
├── <symbol>_<side>_1m.zip
└── <symbol>_<side>_1m_vecnormalize.pkl
```

模型文件和 `VecNormalize` 状态必须配套保存和加载。

### 2. 理解分阶段训练

- Phase 1：在固定数据块上进行基础预热
- Phase 2A：在前若干数据块中随机选择起点和数据块
- Phase 2B：扩展到完整训练数据块池
- `phase2`：依次运行 Phase 2A 和 Phase 2B
- `both`：依次运行所有阶段

如果目标 checkpoint 已存在，训练器会加载模型和归一化状态继续训练。

### 3. 单块对比评估

```bash
python -m src.agents.evaluate \
  --symbol BTCUSDT \
  --side buy \
  --chunk-index 0 \
  --episodes 50 \
  --seed 42 \
  --train-config configs/train_btc_long.yaml \
  --chunk-root /path/to/tardis_chunks
```

评估会在相同的随机起点比较：

- PPO Trained Model
- TWAP Market
- Passive Best Bid/Ask
- Passive Then Sweep

输出包括 reward、成交数量、剩余数量、equity、agent/benchmark cost、excess cost 和 taker fill。

### 4. 多块样本外评估

```bash
python -m src.agents.evaluate_many \
  --symbol BTCUSDT \
  --side buy \
  --train-config configs/train_btc_long.yaml \
  --chunk-root /path/to/tardis_chunks
```

该命令遍历测试日期范围内的所有数据块，输出每块结果以及平均 reward 和标准差。

## 批量训练

交易对列表位于 [`configs/symbol_batch_top5.txt`](configs/symbol_batch_top5.txt)。当前示例包含 APT、LTC、ADA、FIL 和 DOT。

依次运行：

```bash
bash scripts/merge_batch.sh
bash scripts/train_buy_phase1_batch.sh
bash scripts/train_buy_phase2_batch.sh
bash scripts/train_sell_phase1_batch.sh
bash scripts/train_sell_phase2_batch.sh
```

也可以使用：

```bash
bash scripts/nightly_batch.sh
```

注意：这些 shell 脚本当前写死了 `PROJECT_ROOT` 和 `PYTHON_BIN`。在其他机器上运行前，必须将它们改为本机项目路径和 Python 路径。

定时任务示例：

```cron
0 21 * * * cd /path/to/execution_rl_project && flock -n /tmp/exec_rl_batch.lock bash scripts/nightly_batch.sh >> logs/nightly_batch.log 2>&1
```

## 主要配置

- `configs/train_btc_long.yaml`：日期切分、阶段长度、PPO 参数和设备
- `configs/env.yaml`：默认资产、费用、延迟、任务规格和奖励参数
- `configs/env_buy_btc_1m.yaml`：BTC 买入环境示例
- `configs/env_sell_btc_1m.yaml`：BTC 卖出环境示例
- `configs/batch_data.yaml`：本地数据根目录和 train/val/test 日期
- `configs/symbol_batch_top5.txt`：批量训练交易对列表

当前 `ppo_train.py` 会在代码中构造环境配置，并读取训练配置中的奖励参数；它不会直接加载 `env_buy_btc_1m.yaml` 或 `env_sell_btc_1m.yaml`。修改实验参数时，请确认实际入口使用的是哪份配置。

## 实验与复现建议

- 固定随机种子，并记录训练配置、数据日期和 chunk 列表
- 严格分离 train、validation 和 test 日期
- 同时保存 PPO checkpoint 与对应的 VecNormalize 文件
- 使用相同 episode 起点比较 PPO 和基准策略
- 将手续费、延迟、tick size、lot size 和目标数量调整为对应交易对的真实规格
- 批量运行前先用 `--max-chunks 1` 做端到端检查

## 局限与风险

- 回放撮合是现实交易的近似，无法完整还原排队位置、隐藏流动性和网络抖动。
- 当前训练入口对 tick size、lot size、费用和目标数量使用代码内默认值，多资产实验前应逐一校准。
- 历史回测和 RL reward 改善不代表实盘一定能降低成本。
- 本项目仅用于研究，不构成交易或投资建议。

---

# Execution RL Project (English)

A reinforcement-learning execution framework built on high-frequency crypto order-book replay. It uses Tardis Binance Futures incremental book, trade, and snapshot data to construct a Gymnasium environment, then trains Stable-Baselines3 PPO agents to complete buy or sell targets within a fixed horizon.

The current implementation supports multiple symbols, both execution sides, staged training, batch jobs, and matched-start comparisons against TWAP and passive execution baselines.

## Objective

Given an asset, side, target quantity, and deadline, the agent balances execution cost against completion risk:

- Input: order-book state, recent trade features, and task progress
- Output: hold, passive orders, a small market order, or cancellation
- Objective: reduce implementation shortfall while avoiding terminal inventory
- Baselines: TWAP Market, Passive Best Bid/Ask, and Passive Then Sweep

This project addresses order execution rather than signal generation. Target positions may be supplied by the neighboring `cryptoAlpha` project.

## Features

- Parsing for Tardis incremental L2, trades, and 25-level snapshots
- CSV/CSV.GZ conversion into standardized Parquet time chunks
- Custom event-driven order-book replay and fill simulation
- Gymnasium environment for both buy and sell tasks
- PPO training, checkpoint resumption, and observation normalization
- Fixed-chunk warmup, limited-pool random training, and full-pool training
- Single-chunk and multi-chunk out-of-sample evaluation
- Batch scripts for multiple symbols and both execution sides

## Project Structure

```text
execution_rl_project/
├── configs/                         # Environment, data, and training configs
├── scripts/                         # Data preparation, training, and batch jobs
├── src/
│   ├── data/                        # Tardis parsers and sample builders
│   ├── backtest/                    # Order-book replay and execution wrapper
│   ├── features/                    # LOB and task-state features
│   ├── env/                         # Gymnasium environment and rewards
│   ├── baselines/                   # TWAP and passive baselines
│   ├── agents/                      # PPO training, evaluation, and analysis
│   ├── microalpha/                  # Micro-alpha experiments
│   └── utils/                       # Paths, seeds, and chunk utilities
├── results/                         # Models, normalizers, and evaluation results
├── logs/                            # Batch logs created at runtime
└── requirements.txt
```

## Installation

Python 3.10 or later is recommended. Create an isolated environment and install the dependencies:

```bash
cd execution_rl_project
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Core dependencies include Gymnasium, Stable-Baselines3, PyTorch, pandas, NumPy, PyArrow, and HFTBacktest. Training time depends on the dataset size, replay overhead, and hardware. The default configuration uses the CPU.

## Data Preparation

Tardis source data is not included. Each symbol requires Binance Futures incremental L2, trade, and 25-level snapshot files:

```text
incremental_book_L2/<SYMBOL>/<YYYY-MM-DD>/<SYMBOL>.csv[.gz]
trades/<SYMBOL>/<YYYY-MM-DD>/<SYMBOL>.csv[.gz]
book_snapshot_25/<SYMBOL>/<YYYY-MM-DD>/<SYMBOL>.csv[.gz]
```

### 1. Configure Local Paths

Edit [`configs/batch_data.yaml`](configs/batch_data.yaml) and set `local_root` plus the train, validation, and test date ranges. Make sure those dates also agree with [`configs/train_btc_long.yaml`](configs/train_btc_long.yaml).

### 2. Build Standardized Chunks

Convert daily source files directly into the Parquet chunks consumed by training:

```bash
python scripts/build_tardis_chunks.py \
  --symbol BTCUSDT \
  --start-day 2025-12-01 \
  --end-day 2026-01-01 \
  --chunk-hours 6 \
  --raw-root /path/to/raw/tardis \
  --chunk-root /path/to/tardis_chunks
```

Each chunk contains `book.parquet`, `trades.parquet`, `snapshot.parquet`, and `meta.json`. The chunk duration must be a positive divisor of 24 hours. Existing chunks are skipped unless `--force` is supplied.

`src/utils/tardis_chunk.py` contains a machine-specific default chunk root. Pass `--chunk-root` to training and evaluation commands when using a different location.

## Execution Environment

### Action Space

The environment has five discrete actions:

```text
0  HOLD
1  PLACE_PASSIVE_1
2  PLACE_PASSIVE_2
3  MARKET_SMALL
4  CANCEL_ALL
```

For a buy task, the passive actions quote at the best bid and one tick below it. For a sell task, they quote at the best ask and one tick above it. `MARKET_SMALL` executes at most `market_clip_qty` per action.

### Observation Space

The current observation is an eight-dimensional continuous vector: six order-book/recent-trade features plus two execution-progress features. See `src/features/lob_features.py` for the exact construction.

### Reward Modes

- `shortfall`: rewards fills relative to the arrival price and penalizes remaining quantity at the deadline
- TWAP-relative: extends shortfall with TWAP progress, taker-fill, and excess-cost terms

Tune these behaviors with `reward_mode`, `lambda_terminal_remain`, `lambda_lag`, `lambda_taker`, and `lambda_excess` in the training configuration.

## Quick Start

Run all commands from the `execution_rl_project` root.

### 1. Train One PPO Agent

```bash
python -m src.agents.ppo_train \
  --symbol BTCUSDT \
  --side buy \
  --phase both \
  --split train \
  --train-config configs/train_btc_long.yaml \
  --chunk-root /path/to/tardis_chunks
```

Important options:

- `--symbol`: trading pair such as `BTCUSDT`
- `--side`: `buy` or `sell`
- `--phase`: `phase1`, `phase2a`, `phase2b`, `phase2`, or `both`
- `--split`: the configured `train`, `val`, or `test` date range
- `--max-chunks`: load only the first N chunks for quick debugging
- `--chunk-root`: override the default chunk directory

Artifacts are saved under `results/checkpoints_<symbol>_<side>_1m/`. The PPO checkpoint and matching `VecNormalize` file must be preserved and loaded together.

### 2. Staged Training

- Phase 1: warm up on one fixed chunk
- Phase 2A: random starts and chunks from a limited initial pool
- Phase 2B: expand training to the full chunk pool
- `phase2`: run Phase 2A followed by Phase 2B
- `both`: run every phase in sequence

If matching checkpoint files already exist, training resumes from them.

### 3. Compare One Test Chunk

```bash
python -m src.agents.evaluate \
  --symbol BTCUSDT \
  --side buy \
  --chunk-index 0 \
  --episodes 50 \
  --seed 42 \
  --train-config configs/train_btc_long.yaml \
  --chunk-root /path/to/tardis_chunks
```

The command evaluates PPO, TWAP Market, Passive Best Bid/Ask, and Passive Then Sweep from identical sampled starting points. Reported metrics include reward, filled and remaining quantity, equity, agent and benchmark costs, excess cost, and taker fill.

### 4. Evaluate All Test Chunks

```bash
python -m src.agents.evaluate_many \
  --symbol BTCUSDT \
  --side buy \
  --train-config configs/train_btc_long.yaml \
  --chunk-root /path/to/tardis_chunks
```

This command traverses every chunk in the configured test range and reports per-chunk results plus the mean and standard deviation of reward.

## Batch Training

The symbol list lives in [`configs/symbol_batch_top5.txt`](configs/symbol_batch_top5.txt). Run the jobs in order:

```bash
bash scripts/merge_batch.sh
bash scripts/train_buy_phase1_batch.sh
bash scripts/train_buy_phase2_batch.sh
bash scripts/train_sell_phase1_batch.sh
bash scripts/train_sell_phase2_batch.sh
```

Alternatively, execute `bash scripts/nightly_batch.sh`.

The shell scripts currently contain hard-coded `PROJECT_ROOT` and `PYTHON_BIN` values. Update both before running them on another machine.

Example scheduled job:

```cron
0 21 * * * cd /path/to/execution_rl_project && flock -n /tmp/exec_rl_batch.lock bash scripts/nightly_batch.sh >> logs/nightly_batch.log 2>&1
```

## Configuration Reference

- `configs/train_btc_long.yaml`: date splits, stage lengths, PPO settings, and device
- `configs/env.yaml`: default asset, fees, latency, task, and reward settings
- `configs/env_buy_btc_1m.yaml`: example BTC buy environment
- `configs/env_sell_btc_1m.yaml`: example BTC sell environment
- `configs/batch_data.yaml`: local root and train/validation/test dates
- `configs/symbol_batch_top5.txt`: batch symbol universe

`ppo_train.py` currently builds its environment configuration in code and reads reward overrides from the training config. It does not directly load `env_buy_btc_1m.yaml` or `env_sell_btc_1m.yaml`. Always verify which configuration the selected entry point actually consumes.

## Reproducibility Guidance

- Fix random seeds and record configs, data dates, and chunk lists
- Keep train, validation, and test dates strictly separated
- Save the PPO checkpoint together with its `VecNormalize` state
- Compare PPO and baselines using identical episode starts
- Calibrate fees, latency, tick size, lot size, and target quantity for each symbol
- Run an end-to-end smoke test with `--max-chunks 1` before a batch job

## Limitations and Risk

- Replay-based fills approximate real trading and cannot fully reproduce queue position, hidden liquidity, or network jitter.
- The training entry point currently uses in-code defaults for tick size, lot size, fees, and target quantity. Calibrate them before multi-asset experiments.
- Better historical rewards or backtest metrics do not guarantee lower live execution costs.
- This project is for research only and does not constitute trading or investment advice.
