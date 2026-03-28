from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd


def resolve_project_root(start: Path) -> Path:
    current = start.resolve()
    if current.is_file():
        current = current.parent

    while not (current / "src").exists():
        if current.parent == current:
            raise RuntimeError("找不到项目根目录（包含 src 的目录）")
        current = current.parent
    return current


PROJECT_ROOT = resolve_project_root(Path(__file__).resolve())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.factors.factor_builder import FactorBuilder
from src.models.ridge_alpha_model import RidgeAlphaModel
from src.portfolio.portfolio_backtest import (
    backtest_long_short_portfolio,
    build_execution_target_table,
    summarize_portfolio_result,
)


DEFAULT_FACTOR_NAMES = [
    "mom_24h",
    "mom_6h",
    "funding_z_24",
    "oi_change_24h",
    "taker_imbalance",
    "long_short_ratio_z_24",
    "volume_ratio_24",
    "active_community_count_z_24",
]

OUTPUT_START = "2025-12-18"
OUTPUT_END = "2025-12-28 23:59:59"
OUTPUT_TAG = "20251218_20251228"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Ridge alpha training, save outputs, and backtest portfolio.")
    parser.add_argument(
        "--panel-file",
        type=Path,
        default=Path("data/cache/panel_top20_1h_2025_20260311.parquet"),
        help="Parquet panel data path, relative to project root by default.",
    )
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--train-window", type=int, default=180)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--quantile", type=float, default=0.1)
    parser.add_argument("--rebalance-every-hours", type=int, default=1)
    parser.add_argument("--portfolio-forward-hours", type=int, default=1)
    parser.add_argument("--execution-symbol", type=str, default="BTCUSDT")
    parser.add_argument("--execution-start", type=str, default=OUTPUT_START)
    parser.add_argument("--execution-end", type=str, default=OUTPUT_END)
    parser.add_argument("--initial-portfolio-value", type=float, default=1_000_000.0)
    parser.add_argument("--prediction-start", type=str, default=OUTPUT_START)
    parser.add_argument("--prediction-end", type=str, default=OUTPUT_END)
    parser.add_argument(
        "--output-label",
        type=str,
        default="1h",
        help="Suffix used in output directories and filenames, e.g. 1h or 4h.",
    )
    parser.add_argument(
        "--execution-root",
        type=Path,
        default=PROJECT_ROOT.parent / "execution_rl_project",
        help="Path to the RL execution project used for artifact references.",
    )
    return parser.parse_args()


def make_absolute(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def dt_tag(start: str, end: str) -> str:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    return f"{start_ts:%Y%m%d}_{end_ts:%Y%m%d}"


def parse_execution_eval_log(log_path: Path) -> pd.DataFrame:
    lines = log_path.read_text().splitlines()
    meta: dict[str, str] = {"log_path": str(log_path)}
    rows: list[dict[str, object]] = []
    current_row: dict[str, object] | None = None

    for line in lines:
        text = line.strip()
        if not text:
            continue

        if text.startswith("side:"):
            meta["side"] = text.split(":", 1)[1].strip()
            continue

        if text.startswith("chunk:"):
            meta["chunk"] = text.split(":", 1)[1].strip()
            continue

        if text.endswith(":") and not text.startswith("==="):
            current_row = {
                "strategy": text[:-1],
                "side": meta.get("side"),
                "chunk": meta.get("chunk"),
                "log_path": meta["log_path"],
            }
            continue

        if current_row is None:
            continue

        if text.startswith("episodes="):
            current_row["episodes"] = int(text.split("=", 1)[1])
            continue

        match = re.match(r"avg_reward=([-0-9.]+) \| std_reward=([-0-9.]+)", text)
        if match:
            current_row["avg_reward"] = float(match.group(1))
            current_row["std_reward"] = float(match.group(2))
            continue

        match = re.match(r"avg_filled=([-0-9.]+) \| std_filled=([-0-9.]+)", text)
        if match:
            current_row["avg_filled"] = float(match.group(1))
            current_row["std_filled"] = float(match.group(2))
            continue

        match = re.match(r"avg_remaining=([-0-9.]+) \| std_remaining=([-0-9.]+)", text)
        if match:
            current_row["avg_remaining"] = float(match.group(1))
            current_row["std_remaining"] = float(match.group(2))
            continue

        match = re.match(r"avg_equity=([-0-9.]+) \| std_equity=([-0-9.]+)", text)
        if match:
            current_row["avg_equity"] = float(match.group(1))
            current_row["std_equity"] = float(match.group(2))
            rows.append(current_row.copy())

    return pd.DataFrame(rows)


def build_chunk_list(execution_symbol: str, execution_dir: Path, target_days: set[str], output_tag: str) -> pd.DataFrame:
    chunk_root = Path("/home/joyce/projects/data/raw/tardis_chunks") / execution_symbol.upper()
    chunk_records: list[dict[str, object]] = []

    if not chunk_root.exists():
        return pd.DataFrame(columns=["chunk_index", "chunk", "book_path", "trade_path", "snapshot_path"])

    for chunk_index, chunk_dir in enumerate(sorted(p for p in chunk_root.iterdir() if p.is_dir())):
        chunk_name = chunk_dir.name
        if chunk_name[:10] in target_days:
            chunk_records.append(
                {
                    "chunk_index": chunk_index,
                    "chunk": chunk_name,
                    "book_path": str(chunk_dir / "book.parquet"),
                    "trade_path": str(chunk_dir / "trades.parquet"),
                    "snapshot_path": str(chunk_dir / "snapshot.parquet"),
                }
            )

    chunk_list_df = pd.DataFrame(chunk_records)
    chunk_list_df.to_csv(execution_dir / f"btc_chunk_list_{output_tag}.csv", index=False)
    return chunk_list_df


def build_execution_artifact_manifest(execution_root: Path, execution_dir: Path, output_tag: str) -> pd.DataFrame:
    artifact_records = [
        {
            "artifact": "train_config",
            "path": str(execution_root / "configs" / "train_btc_long.yaml"),
        },
        {
            "artifact": "buy_checkpoint",
            "path": str(execution_root / "results" / "checkpoints_btcusdt_buy_1m" / "btcusdt_buy_1m.zip"),
        },
        {
            "artifact": "buy_vecnormalize",
            "path": str(execution_root / "results" / "checkpoints_btcusdt_buy_1m" / "btcusdt_buy_1m_vecnormalize.pkl"),
        },
        {
            "artifact": "sell_checkpoint",
            "path": str(execution_root / "results" / "checkpoints_btcusdt_sell_1m" / "btcusdt_sell_1m.zip"),
        },
        {
            "artifact": "sell_vecnormalize",
            "path": str(execution_root / "results" / "checkpoints_btcusdt_sell_1m" / "btcusdt_sell_1m_vecnormalize.pkl"),
        },
        {
            "artifact": "buy_train_log",
            "path": str(execution_root / ".copilot_run_logs" / "btcusdt_buy" / "step3_train_both.log"),
        },
        {
            "artifact": "sell_train_log",
            "path": str(execution_root / ".copilot_run_logs" / "btcusdt_sell" / "step3_train_both.log"),
        },
    ]
    artifact_df = pd.DataFrame(artifact_records)
    artifact_df["exists"] = artifact_df["path"].map(lambda x: Path(x).exists())
    artifact_df.to_csv(execution_dir / f"execution_artifact_manifest_{output_tag}.csv", index=False)
    return artifact_df


def build_execution_evaluation_table(execution_root: Path, execution_dir: Path, output_tag: str) -> pd.DataFrame:
    eval_log_paths = [
        execution_root / ".copilot_run_logs" / "btcusdt_buy" / "eval_buy_2025-12-26_18_50episodes.log",
        execution_root / ".copilot_run_logs" / "btcusdt_buy" / "eval_buy_2025-12-27_18_20episodes.log",
        execution_root / ".copilot_run_logs" / "btcusdt_buy" / "eval_buy_2025-12-28_00_20episodes.log",
        execution_root / ".copilot_run_logs" / "btcusdt_sell" / "eval_sell_2025-12-26_18_20episodes.log",
        execution_root / ".copilot_run_logs" / "btcusdt_sell" / "eval_sell_2025-12-27_18_20episodes.log",
        execution_root / ".copilot_run_logs" / "btcusdt_sell" / "eval_sell_2025-12-28_00_20episodes.log",
    ]

    frames = [parse_execution_eval_log(path) for path in eval_log_paths if path.exists()]
    if frames:
        execution_eval_df = pd.concat(frames, ignore_index=True)
    else:
        execution_eval_df = pd.DataFrame()
    execution_eval_df.to_csv(execution_dir / f"execution_evaluation_{output_tag}.csv", index=False)
    return execution_eval_df


def main() -> None:
    args = parse_args()
    output_tag = dt_tag(args.prediction_start, args.prediction_end)
    output_label = args.output_label.strip()

    panel_fp = make_absolute(args.panel_file)
    pred_dir = ensure_dir(PROJECT_ROOT / "data" / f"predictions_{output_label}")
    model_dir = ensure_dir(PROJECT_ROOT / "data" / "models" / output_label)
    execution_dir = ensure_dir(PROJECT_ROOT / "data" / f"execution_{output_label}")
    backtest_dir = ensure_dir(PROJECT_ROOT / "data" / f"backtest_{output_label}")

    print(f"PROJECT_ROOT = {PROJECT_ROOT}")
    print(f"panel_fp = {panel_fp}")
    print(f"output_label = {output_label}")

    panel_df = pd.read_parquet(panel_fp)
    prediction_start_ts = pd.Timestamp(args.prediction_start)
    prediction_end_ts = pd.Timestamp(args.prediction_end)

    print("[1/6] computing factors")
    builder = FactorBuilder()
    factor_df = builder.compute_many(panel_df, DEFAULT_FACTOR_NAMES)
    factor_df[
        factor_df["datetime"].between(prediction_start_ts, prediction_end_ts)
    ].to_parquet(backtest_dir / f"ridge_factor_df_{output_label}_{output_tag}.parquet", index=False)

    print("[2/6] training ridge alpha and saving daily models")
    model = RidgeAlphaModel(
        horizon=args.horizon,
        train_window=args.train_window,
        alpha=args.alpha,
    )
    df_model = model.build_label(panel_df)
    df_model = df_model.merge(
        factor_df,
        on=["datetime", "symbol"],
        how="left",
    )

    pred_df = model.fit_predict_save_daily_models(
        df=df_model,
        feature_cols=DEFAULT_FACTOR_NAMES,
        daily_model_dir=model_dir,
        label_col="future_return",
    )

    pred_df = pred_df[
        pred_df["datetime"].between(prediction_start_ts, prediction_end_ts)
    ].reset_index(drop=True)

    for model_fp in model_dir.glob("ridge_alpha_*.pkl"):
        stem = model_fp.stem.replace("ridge_alpha_", "")
        try:
            model_ts = pd.to_datetime(stem, format="%Y%m%d_%H%M")
        except ValueError:
            continue
        if not (prediction_start_ts <= model_ts <= prediction_end_ts):
            model_fp.unlink(missing_ok=True)

    pred_fp = pred_dir / f"pred_df_top20_1h_{output_label}_{output_tag}_ridge.parquet"
    pred_df.to_parquet(pred_fp, index=False)

    print("[3/6] running backtest")
    bt_result = backtest_long_short_portfolio(
        pred_df=pred_df,
        panel_df=panel_df,
        quantile=args.quantile,
        rebalance_every_hours=args.rebalance_every_hours,
        portfolio_forward_hours=args.portfolio_forward_hours,
    )
    summary = summarize_portfolio_result(bt_result)

    pd.DataFrame([summary]).to_csv(backtest_dir / f"ridge_portfolio_summary_{output_label}_{output_tag}.csv", index=False)
    bt_result["weight_df"].to_parquet(backtest_dir / f"ridge_weight_df_{output_label}_{output_tag}.parquet", index=False)
    bt_result["holding_df"].to_parquet(backtest_dir / f"ridge_holding_df_{output_label}_{output_tag}.parquet", index=False)
    bt_result["portfolio_return"].rename("portfolio_return").to_csv(
        backtest_dir / f"ridge_portfolio_return_{output_label}_{output_tag}.csv"
    )
    bt_result["cumret"].rename("cumret").to_csv(backtest_dir / f"ridge_cumret_{output_label}_{output_tag}.csv")

    print("[4/6] building BTC execution target table")
    btc_execution_df = build_execution_target_table(
        bt_result=bt_result,
        panel_df=panel_df,
        symbol=args.execution_symbol,
        initial_portfolio_value=args.initial_portfolio_value,
    )
    btc_execution_window_df = (
        btc_execution_df[
            btc_execution_df["datetime"].between(args.execution_start, args.execution_end)
        ]
        .rename(
            columns={
                "current_weight": "BTC current_weight",
                "target_weight": "BTC target_weight",
            }
        )
        .reset_index(drop=True)
    )
    btc_execution_window_df.to_csv(execution_dir / f"btc_execution_targets_{output_label}_{output_tag}.csv", index=False)

    print("[5/6] collecting execution reference tables")
    target_days = {
        d.strftime("%Y-%m-%d")
        for d in pd.date_range(start=prediction_start_ts.normalize(), end=prediction_end_ts.normalize(), freq="D")
    }
    build_chunk_list(args.execution_symbol, execution_dir, target_days, f"{output_label}_{output_tag}")
    build_execution_artifact_manifest(args.execution_root, execution_dir, f"{output_label}_{output_tag}")
    build_execution_evaluation_table(args.execution_root, execution_dir, f"{output_label}_{output_tag}")

    print("[6/6] saving pipeline metadata")
    metadata = {
        "panel_fp": str(panel_fp),
        "prediction_fp": str(pred_fp),
        "daily_model_dir": str(model_dir),
        "backtest_dir": str(backtest_dir),
        "execution_dir": str(execution_dir),
        "factor_names": DEFAULT_FACTOR_NAMES,
        "summary": summary,
        "config": {
            "horizon": args.horizon,
            "train_window": args.train_window,
            "alpha": args.alpha,
            "quantile": args.quantile,
            "rebalance_every_hours": args.rebalance_every_hours,
            "portfolio_forward_hours": args.portfolio_forward_hours,
            "execution_symbol": args.execution_symbol,
            "execution_start": args.execution_start,
            "execution_end": args.execution_end,
            "initial_portfolio_value": args.initial_portfolio_value,
        },
    }
    (backtest_dir / f"ridge_pipeline_metadata_{output_label}_{output_tag}.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str)
    )

    print("done")
    print(f"saved predictions to: {pred_fp}")
    print(f"saved backtest summary to: {backtest_dir / f'ridge_portfolio_summary_{output_label}_{output_tag}.csv'}")
    print(f"saved execution table to: {execution_dir / f'btc_execution_targets_{output_label}_{output_tag}.csv'}")


if __name__ == "__main__":
    main()