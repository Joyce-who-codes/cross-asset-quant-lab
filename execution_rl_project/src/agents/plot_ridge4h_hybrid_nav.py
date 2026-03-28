from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.utils.project_paths import PROJECT_ROOT, results_root


DEFAULT_BACKTEST_CUMRET = (
    PROJECT_ROOT.parent / "cryptoAlpha" / "data" / "backtest_4h" / "ridge_cumret_4h_20251218_20251228.csv"
)
DEFAULT_RL_REPLAY = (
    PROJECT_ROOT / "results" / "ridge4h_rl_bridge" / "20251218_20251228" / "btc_ridge4h_rl_replay_20251218_20251228.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "ridge4h_rl_bridge" / "20251218_20251228"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot baseline Ridge NAV vs hybrid BTC-RL NAV")
    parser.add_argument("--baseline-cumret", type=str, default=str(DEFAULT_BACKTEST_CUMRET))
    parser.add_argument("--rl-replay", type=str, default=str(DEFAULT_RL_REPLAY))
    parser.add_argument("--start", type=str, default="2025-12-18")
    parser.add_argument("--end", type=str, default="2025-12-28 23:59:59")
    parser.add_argument("--initial-nav", type=float, default=1_000_000.0)
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def load_baseline(path: Path, start: str, end: str, initial_nav: float) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    out = df[(df["datetime"] >= start_ts) & (df["datetime"] <= end_ts)].copy()
    out = out.sort_values("datetime").reset_index(drop=True)
    out["baseline_nav"] = out["cumret"].astype(float) * float(initial_nav)
    return out


def load_rl_replay(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["signal_datetime"] = pd.to_datetime(df["signal_datetime"], utc=True)
    df = df.sort_values("signal_datetime").reset_index(drop=True)

    df["direct_notional"] = df["abs_target_qty"].astype(float) * df["signal_price"].astype(float)
    df["rl_notional"] = df["ppo_agent_total_cost"].astype(float)
    df["execution_adjustment"] = df.apply(
        lambda row: row["direct_notional"] - row["rl_notional"]
        if row["side"] == "buy"
        else row["rl_notional"] - row["direct_notional"],
        axis=1,
    )
    df["cum_execution_adjustment"] = df["execution_adjustment"].cumsum()
    return df


def build_nav_comparison(
    baseline_df: pd.DataFrame,
    replay_df: pd.DataFrame,
    initial_nav: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    trade_adj = replay_df[["signal_datetime", "side", "abs_target_qty", "signal_price", "ppo_arrival_price", "direct_notional", "rl_notional", "execution_adjustment", "cum_execution_adjustment"]].copy()
    trade_adj = trade_adj.rename(columns={"signal_datetime": "datetime"})

    timeline = baseline_df[["datetime", "cumret", "baseline_nav"]].copy()
    timeline = timeline.merge(
        trade_adj[["datetime", "execution_adjustment", "cum_execution_adjustment"]],
        on="datetime",
        how="left",
    )
    timeline["execution_adjustment"] = timeline["execution_adjustment"].fillna(0.0)
    timeline["cum_execution_adjustment"] = timeline["cum_execution_adjustment"].ffill().fillna(0.0)
    timeline["hybrid_nav"] = timeline["baseline_nav"] + timeline["cum_execution_adjustment"]
    timeline["hybrid_cumret"] = timeline["hybrid_nav"] / float(initial_nav)
    return timeline, trade_adj


def plot_nav(timeline: pd.DataFrame, replay_df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(timeline["datetime"], timeline["baseline_nav"], label="Control: Ridge low-freq direct execution", linewidth=2)
    plt.plot(timeline["datetime"], timeline["hybrid_nav"], label="Experiment: BTC via RL, others direct", linewidth=2)

    if len(replay_df):
        trade_times = replay_df["signal_datetime"]
        trade_navs = timeline.set_index("datetime").reindex(trade_times, method="nearest")["hybrid_nav"]
        plt.scatter(trade_times, trade_navs, s=30, marker="o", label="BTC RL trades")

    plt.title("Ridge 4h NAV Comparison: Baseline vs BTC-RL Hybrid")
    plt.xlabel("Datetime")
    plt.ylabel("NAV")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    baseline_path = Path(args.baseline_cumret)
    rl_replay_path = Path(args.rl_replay)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_df = load_baseline(
        path=baseline_path,
        start=args.start,
        end=args.end,
        initial_nav=args.initial_nav,
    )
    replay_df = load_rl_replay(rl_replay_path)
    timeline, trade_adj = build_nav_comparison(
        baseline_df=baseline_df,
        replay_df=replay_df,
        initial_nav=args.initial_nav,
    )

    timeline_csv = output_dir / "btc_rl_hybrid_nav_comparison_20251218_20251228.csv"
    trade_csv = output_dir / "btc_rl_trade_adjustments_20251218_20251228.csv"
    plot_path = output_dir / "btc_rl_hybrid_nav_comparison_20251218_20251228.png"
    summary_path = output_dir / "btc_rl_hybrid_nav_comparison_20251218_20251228_summary.json"

    timeline.to_csv(timeline_csv, index=False)
    trade_adj.to_csv(trade_csv, index=False)
    plot_nav(timeline=timeline, replay_df=replay_df, output_path=plot_path)

    summary = {
        "baseline_path": str(baseline_path),
        "rl_replay_path": str(rl_replay_path),
        "timeline_csv": str(timeline_csv),
        "trade_adjustment_csv": str(trade_csv),
        "plot_path": str(plot_path),
        "initial_nav": float(args.initial_nav),
        "baseline_final_nav": float(timeline["baseline_nav"].iloc[-1]),
        "hybrid_final_nav": float(timeline["hybrid_nav"].iloc[-1]),
        "final_nav_diff": float(timeline["hybrid_nav"].iloc[-1] - timeline["baseline_nav"].iloc[-1]),
        "total_execution_adjustment": float(trade_adj["execution_adjustment"].sum()),
        "num_btc_rl_trades": int(len(trade_adj)),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
