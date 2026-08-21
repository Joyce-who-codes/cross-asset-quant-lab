# cryptoAlpha

一个面向加密资产的跨品种 Alpha 研究框架，用于将多源行情、衍生品和情绪数据整理为统一面板，完成因子构建、单因子检验、滚动模型训练与截面多空组合回测。

当前项目主要围绕 Top 20 加密资产的小时级研究展开，并支持将组合目标导出给相邻的 `execution_rl_project` 执行层项目。

## 主要功能

- 多源数据接入：CoinGlass、CoinGecko、CryptOracle
- 统一数据结构：以 `datetime` 和 `symbol` 为主键构建跨资产面板
- 因子研究：价格动量、波动率、成交量/流动性、衍生品和情绪类因子
- 因子评价：IC、Rank IC、分组收益、Top-Bottom 收益、换手率、覆盖率和年度统计
- Alpha 建模：Ridge 与 XGBoost 滚动窗口训练及预测
- 组合回测：按预测分位数构建市场中性多空组合
- 结果输出：Parquet/CSV 数据、模型文件、统计摘要和可视化报告

## 项目结构

```text
cryptoAlpha/
├── configs/                         # 数据路径、因子、模型与回测配置
├── factor_lib/expressions/          # 因子表达式定义（YAML）
├── src/
│   ├── data/
│   │   ├── loaders/                 # 各数据源加载器
│   │   ├── eda/                     # 数据源画像与低频数据报告
│   │   ├── panel_builder.py         # 多源数据面板合并工具
│   │   ├── registry.py              # 数据路径注册与解析
│   │   └── schema.py                # 字段标准化
│   ├── factors/                     # 因子计算与因子注册表
│   ├── evaluation/                  # 单因子及多周期评价
│   ├── models/                      # Ridge、XGBoost Alpha 模型
│   ├── portfolio/                   # 组合构建、回测与绘图
│   ├── backtest/                    # 通用回测引擎与指标
│   └── analysis/                    # 可直接运行的研究流程与 notebooks
├── data/                            # 本地数据与中间产物（运行后生成）
├── outputs/                         # 配置中约定的模型/回测输出目录
└── requirements.txt
```

## 环境安装

建议使用 Python 3.10 或以上版本，并在独立虚拟环境中安装依赖。

```bash
cd cryptoAlpha
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

XGBoost 流程和模型持久化还会使用 `xgboost` 与 `joblib`。如果当前环境未安装，请额外执行：

```bash
python -m pip install xgboost joblib
```

## 数据准备

项目不附带原始数据。运行前需要编辑 [`configs/paths.yaml`](configs/paths.yaml)，将三个数据源目录改为本机的实际路径：

```yaml
sources:
  coinglass_root: /path/to/coinglass
  coingecko_root: /path/to/coingecko
  cryptoracle_root: /path/to/cryptoracle
```

默认频率为 `1h`，默认交易所为 `Binance`。存储目录也可以在同一配置中调整。

标准面板至少需要以下主键：

- `datetime`：时间戳；内部会标准化为无时区的 pandas datetime
- `symbol`：资产代码，例如 `BTCUSDT`

价格类研究通常还需要 `open`、`high`、`low`、`close` 和 `volume_usd`。衍生品及情绪因子会按需读取 `funding_close`、`oi_close`、主动买卖成交量、多空账户比例和社区活跃度等字段。缺少某个因子所需字段时，该因子无法计算。

现有分析脚本默认读取：

```text
data/cache/panel_top20_1h_2025_20260311.parquet
```

也可以通过命令行参数传入其他面板文件。

## 快速开始

以下命令均从 `cryptoAlpha` 根目录执行。

### 1. 查看可用因子

```bash
python -c "from src.factors.factor_builder import FactorBuilder; print(FactorBuilder().list_factors())"
```

因子函数及注册表位于 `src/factors/factor_builder.py`。新增因子时，需要实现返回 `datetime`、`symbol` 和因子值的函数，并将其加入 `FACTOR_REGISTRY`。

### 2. 批量运行单因子检验

```bash
python src/analysis/run_all_single_factor_tests.py \
  --panel-file data/cache/panel_top20_1h_2025_20260311.parquet \
  --output-dir data/evaluation/single_factor_all_1h \
  --horizon 1 \
  --group-num 5 \
  --price-col close
```

该流程会计算注册表中的全部因子，并输出：

- 全因子汇总表与失败记录
- 每个因子的 IC、Rank IC 和分组收益序列
- Top-Bottom 收益、覆盖率、换手率与年度 IC
- 单因子完整报告图及全局对比图

### 3. 运行 Ridge Alpha 与组合回测

```bash
python src/analysis/portfolio_ridge_pipeline.py \
  --panel-file data/cache/panel_top20_1h_2025_20260311.parquet \
  --horizon 1 \
  --train-window 180 \
  --alpha 1.0 \
  --quantile 0.1 \
  --rebalance-every-hours 1 \
  --portfolio-forward-hours 1 \
  --prediction-start 2025-12-18 \
  --prediction-end 2025-12-28T23:59:59 \
  --output-label 1h
```

主要产物会写入：

```text
data/predictions_1h/   # Alpha 分数
data/models/1h/        # 滚动训练的 Ridge 模型
data/backtest_1h/      # 持仓、权重、收益、净值与统计摘要
data/execution_1h/     # 执行层目标及相关产物清单
```

脚本默认选用以下八个因子：

```text
mom_24h, mom_6h, funding_z_24, oi_change_24h,
taker_imbalance, long_short_ratio_z_24,
volume_ratio_24, active_community_count_z_24
```

如果面板缺少其中任意输入字段，请先补齐数据，或在脚本的 `DEFAULT_FACTOR_NAMES` 中调整因子集合。

### 4. 运行 XGBoost Alpha 示例

```bash
python src/analysis/portfolio_run.py
```

该脚本目前使用代码内固定的面板路径、因子集合和训练参数，适合作为研究模板。若要用于新的数据区间，请先修改脚本顶部的配置。

### 5. 导出组合回测报告

在 Ridge 流程生成回测产物后，可以执行：

```bash
python src/analysis/export_portfolio_backtest_report.py \
  --backtest-dir data/backtest_1h \
  --output-label 1h \
  --date-tag 20251218_20251228
```

`date-tag` 需要与预测起止日期生成的标签一致。

## 核心研究流程

```text
CoinGlass / CoinGecko / CryptOracle
                  │
                  ▼
        字段标准化与时间对齐
                  │
                  ▼
      (datetime, symbol) 统一面板
                  │
          ┌───────┴────────┐
          ▼                ▼
      单因子评价        多因子模型训练
                           │
                           ▼
                      Alpha score
                           │
                           ▼
                    截面多空组合回测
                           │
                           ▼
                  研究报告 / 执行目标
```

模型采用滚动窗口训练，避免直接使用未来数据。标签由每个资产未来 `horizon` 个周期的收益构造；组合层根据预测分数选择顶部和底部分位资产，生成多空权重并计算组合表现。

## 配置说明

- `configs/paths.yaml`：实际数据源路径、默认频率和本地存储路径
- `configs/data.yaml`：原始数据、处理后数据、特征、标签和缓存目录
- `configs/factors.yaml`：启用的因子集合及因子表输出位置
- `configs/model.yaml`：模型类型、标签字段、特征表和模型输出目录
- `configs/backtest.yaml`：再平衡频率、交易成本和最大杠杆
- `configs/universe.yaml`：研究资产池（当前文件保留 Top 20 示例）

注意：目前并非所有分析脚本都会自动读取所有 YAML 配置，部分脚本仍使用代码内默认值。正式批量运行前，请同时核对命令行参数和脚本顶部常量。

## 输出与版本控制建议

原始数据、缓存、模型文件和回测结果通常体积较大，建议不要直接提交到 Git。可重点保留：

- 配置文件和因子定义
- 汇总指标及小型报告
- 可复现研究所需的脚本与参数
- 数据版本、时间区间和来源说明

## 研究注意事项

- 本项目用于量化研究，不构成投资建议。
- 回测结果对数据质量、幸存者偏差、交易成本和成交假设非常敏感。
- 在扩展到实盘前，应检查时间戳对齐、数据延迟、异常值、缺失值和前视偏差。
- 当前组合回测与真实撮合之间仍有差异；实际执行成本可结合 `execution_rl_project` 进一步评估。

---

# cryptoAlpha (English)

A cross-sectional crypto Alpha research framework for transforming market, derivatives, and sentiment data from multiple sources into a unified panel, then performing factor construction, single-factor evaluation, rolling model training, and long-short portfolio backtesting.

The current research workflow focuses on an hourly Top 20 crypto universe. It can also export portfolio targets for the neighboring `execution_rl_project` execution layer.

## Features

- Multi-source ingestion from CoinGlass, CoinGecko, and CryptOracle
- A unified cross-asset panel keyed by `datetime` and `symbol`
- Price momentum, volatility, volume/liquidity, derivatives, and sentiment factors
- IC, Rank IC, quantile return, top-minus-bottom return, turnover, coverage, and yearly evaluations
- Rolling Ridge and XGBoost Alpha modeling
- Market-neutral long-short portfolio construction based on score quantiles
- Parquet/CSV artifacts, model files, statistical summaries, and visual reports

## Project Structure

```text
cryptoAlpha/
├── configs/                         # Data path, factor, model, and backtest configs
├── factor_lib/expressions/          # Factor expressions in YAML
├── src/
│   ├── data/
│   │   ├── loaders/                 # Data-source loaders
│   │   ├── eda/                     # Source profiling and low-frequency reports
│   │   ├── panel_builder.py         # Multi-source panel merge utilities
│   │   ├── registry.py              # Data path registry and resolution
│   │   └── schema.py                # Schema standardization
│   ├── factors/                     # Factor calculations and registry
│   ├── evaluation/                  # Single-factor and multi-horizon evaluation
│   ├── models/                      # Ridge and XGBoost Alpha models
│   ├── portfolio/                   # Portfolio construction, backtesting, and plots
│   ├── backtest/                    # General backtest engine and metrics
│   └── analysis/                    # Executable research pipelines and notebooks
├── data/                            # Local data and generated artifacts
├── outputs/                         # Configured model and backtest outputs
└── requirements.txt
```

## Installation

Python 3.10 or later is recommended. Install the dependencies in an isolated virtual environment:

```bash
cd cryptoAlpha
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The XGBoost workflow and model serialization also require `xgboost` and `joblib`. Install them separately if they are not already available:

```bash
python -m pip install xgboost joblib
```

## Data Preparation

Raw datasets are not included. Before running the project, edit [`configs/paths.yaml`](configs/paths.yaml) and replace the source roots with paths available on your machine:

```yaml
sources:
  coinglass_root: /path/to/coinglass
  coingecko_root: /path/to/coingecko
  cryptoracle_root: /path/to/cryptoracle
```

The default frequency is `1h`, and the default exchange is `Binance`. Local storage directories can be changed in the same configuration file.

A standardized panel must contain at least:

- `datetime`: timestamps, normalized internally to timezone-naive pandas datetimes
- `symbol`: asset identifiers such as `BTCUSDT`

Price-based research generally requires `open`, `high`, `low`, `close`, and `volume_usd`. Derivatives and sentiment factors may additionally require fields such as `funding_close`, `oi_close`, taker buy/sell volume, long-short account ratios, and community activity. A factor cannot be computed if its required input columns are missing.

The current analysis scripts use the following panel by default:

```text
data/cache/panel_top20_1h_2025_20260311.parquet
```

You can provide a different panel through the available command-line arguments.

## Quick Start

Run the following commands from the `cryptoAlpha` root directory.

### 1. List Available Factors

```bash
python -c "from src.factors.factor_builder import FactorBuilder; print(FactorBuilder().list_factors())"
```

Factor functions and the registry are defined in `src/factors/factor_builder.py`. To add a factor, implement a function that returns `datetime`, `symbol`, and the factor value, then add it to `FACTOR_REGISTRY`.

### 2. Run All Single-Factor Tests

```bash
python src/analysis/run_all_single_factor_tests.py \
  --panel-file data/cache/panel_top20_1h_2025_20260311.parquet \
  --output-dir data/evaluation/single_factor_all_1h \
  --horizon 1 \
  --group-num 5 \
  --price-col close
```

This pipeline computes every registered factor and produces:

- A global factor summary and a failure log
- IC, Rank IC, and quantile return series for each factor
- Top-minus-bottom returns, coverage, turnover, and yearly IC statistics
- Full per-factor reports and global comparison charts

### 3. Run the Ridge Alpha and Portfolio Pipeline

```bash
python src/analysis/portfolio_ridge_pipeline.py \
  --panel-file data/cache/panel_top20_1h_2025_20260311.parquet \
  --horizon 1 \
  --train-window 180 \
  --alpha 1.0 \
  --quantile 0.1 \
  --rebalance-every-hours 1 \
  --portfolio-forward-hours 1 \
  --prediction-start 2025-12-18 \
  --prediction-end 2025-12-28T23:59:59 \
  --output-label 1h
```

The main artifacts are written to:

```text
data/predictions_1h/   # Alpha scores
data/models/1h/        # Rolling Ridge models
data/backtest_1h/      # Holdings, weights, returns, NAV, and summaries
data/execution_1h/     # Execution targets and artifact manifests
```

The pipeline uses these eight factors by default:

```text
mom_24h, mom_6h, funding_z_24, oi_change_24h,
taker_imbalance, long_short_ratio_z_24,
volume_ratio_24, active_community_count_z_24
```

If the panel does not contain the required input fields, either add the missing data or update `DEFAULT_FACTOR_NAMES` in the pipeline script.

### 4. Run the XGBoost Alpha Example

```bash
python src/analysis/portfolio_run.py
```

This script currently uses hard-coded panel paths, factors, and training parameters. Treat it as a research template and update the constants near the top of the script before using a different dataset or date range.

### 5. Export a Portfolio Backtest Report

After generating Ridge backtest artifacts, run:

```bash
python src/analysis/export_portfolio_backtest_report.py \
  --backtest-dir data/backtest_1h \
  --output-label 1h \
  --date-tag 20251218_20251228
```

The `date-tag` must match the tag generated from the prediction start and end dates.

## Research Workflow

```text
CoinGlass / CoinGecko / CryptOracle
                  │
                  ▼
       Schema and timestamp alignment
                  │
                  ▼
      Unified (datetime, symbol) panel
                  │
          ┌───────┴────────┐
          ▼                ▼
   Factor evaluation   Multi-factor model
                           │
                           ▼
                       Alpha score
                           │
                           ▼
                Cross-sectional portfolio
                           │
                           ▼
                Reports / execution targets
```

Models are trained with rolling windows to avoid directly using future observations. Labels are defined as each asset's return over the next `horizon` periods. The portfolio layer selects assets from the top and bottom score quantiles, assigns long-short weights, and evaluates the resulting portfolio.

## Configuration Reference

- `configs/paths.yaml`: source roots, default frequency, and local storage paths
- `configs/data.yaml`: raw, processed, feature, label, and cache directories
- `configs/factors.yaml`: enabled factor sets and factor-table output path
- `configs/model.yaml`: model type, label, feature table, and model output path
- `configs/backtest.yaml`: rebalance frequency, transaction costs, and maximum leverage
- `configs/universe.yaml`: research universe, currently containing a Top 20 example

Not every analysis script automatically consumes every YAML configuration. Some scripts still use in-code defaults. Check both command-line options and constants near the top of each script before starting a production research run.

## Output and Version-Control Guidance

Raw datasets, caches, model files, and backtest outputs can be large and generally should not be committed directly to Git. Prefer retaining:

- Configuration files and factor definitions
- Summary metrics and compact reports
- Scripts and parameters needed for reproducibility
- Data versions, date ranges, and source documentation

## Research Caveats

- This project is intended for quantitative research and does not constitute investment advice.
- Backtest results are highly sensitive to data quality, survivorship bias, transaction costs, and fill assumptions.
- Check timestamp alignment, data availability delays, outliers, missing values, and look-ahead bias before extending the workflow to live trading.
- Portfolio backtests do not fully reproduce real matching behavior. Use `execution_rl_project` to study practical execution costs in greater detail.
