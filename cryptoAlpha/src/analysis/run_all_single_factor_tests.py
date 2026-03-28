from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
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

from src.evaluation.factor_evaluator import FactorEvaluator
from src.factors.factor_builder import FactorBuilder


DEFAULT_PANEL_FP = PROJECT_ROOT / "data" / "cache" / "panel_top20_1h_2025_20260311.parquet"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "evaluation" / "single_factor_all_1h"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single-factor tests for all registered factors and save unified tables/plots.")
    parser.add_argument("--panel-file", type=Path, default=DEFAULT_PANEL_FP)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--group-num", type=int, default=5)
    parser.add_argument("--price-col", type=str, default="close")
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_series_csv(series: pd.Series, path: Path, value_name: str) -> None:
    df = series.rename(value_name).reset_index()
    df.to_csv(path, index=False)


def plot_full_report_to_file(
    result: dict,
    factor_name: str,
    output_path: Path,
    figsize: tuple[int, int] = (16, 18),
) -> None:
    fig, axes = plt.subplots(4, 2, figsize=figsize)
    axes = axes.flatten()

    qcum = result["quantile_returns"].cumsum()
    qcum.plot(ax=axes[0], title=f"{factor_name} Quantile Cumulative Returns")
    axes[0].axhline(0, linestyle="--", color="gray", linewidth=1)

    result["top_bottom_cumret"].plot(ax=axes[1], title=f"{factor_name} Top-Bottom Cumulative Return")
    axes[1].axhline(0, linestyle="--", color="gray", linewidth=1)

    result["ic_series"].plot(ax=axes[2], title=f"{factor_name} IC Series")
    axes[2].axhline(0, linestyle="--", color="gray", linewidth=1)

    result["rank_ic_series"].plot(ax=axes[3], title=f"{factor_name} RankIC Series")
    axes[3].axhline(0, linestyle="--", color="gray", linewidth=1)

    result["ic_series"].dropna().hist(ax=axes[4], bins=40)
    axes[4].set_title(f"{factor_name} IC Histogram")

    result["factor_autocorr"].plot(ax=axes[5], title=f"{factor_name} Factor Autocorrelation")
    axes[5].axhline(0, linestyle="--", color="gray", linewidth=1)

    result["coverage_by_date"].plot(ax=axes[6], title=f"{factor_name} Coverage by Date")
    axes[6].axhline(result["coverage_by_date"].mean(), linestyle="--", color="gray", linewidth=1)

    turnover_df = result["group_turnover"]
    if not turnover_df.empty:
        pivot_to = turnover_df.pivot(index="datetime", columns="quantile", values="turnover")
        pivot_to.plot(ax=axes[7], title=f"{factor_name} Quantile Turnover")
    else:
        axes[7].set_title(f"{factor_name} Quantile Turnover (No Data)")

    plt.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_quantile_bar_to_file(
    result: dict,
    factor_name: str,
    output_path: Path,
    figsize: tuple[int, int] = (10, 4),
) -> None:
    mean_ret = result["quantile_returns"].mean()
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    mean_ret.plot(kind="bar", ax=ax)
    ax.set_title(f"{factor_name} Mean Return by Quantile")
    ax.set_ylabel("Mean Forward Return")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_yearly_ic_table_to_file(
    result: dict,
    factor_name: str,
    output_path: Path,
) -> None:
    yearly = result["yearly_ic_table"].copy()
    fig, ax = plt.subplots(figsize=(12, max(2, 0.5 * len(yearly) + 1)))
    ax.axis("off")
    if yearly.empty:
        ax.text(0.5, 0.5, f"{factor_name}: no yearly IC data", ha="center", va="center")
    else:
        show_df = yearly.copy()
        for col in show_df.columns:
            if col != "year":
                show_df[col] = show_df[col].map(lambda x: f"{x:.4f}")
        table = ax.table(
            cellText=show_df.values,
            colLabels=show_df.columns,
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.2)
        ax.set_title(f"{factor_name} Yearly IC Table", pad=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_global_summary(summary_df: pd.DataFrame, output_dir: Path) -> None:
    top = summary_df.sort_values("rank_ic_mean", ascending=False)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(top["factor_name"], top["rank_ic_mean"], color="#4C78A8")
    ax.set_title("Rank IC mean by factor")
    ax.set_ylabel("Rank IC mean")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "summary_rank_ic_mean.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(top["factor_name"], top["top_bottom_sharpe_naive"], color="#F58518")
    ax.set_title("Naive top-bottom Sharpe by factor")
    ax.set_ylabel("Top-bottom Sharpe")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "summary_top_bottom_sharpe.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(top["rank_ic_mean"], top["top_bottom_sharpe_naive"], color="#54A24B")
    for _, row in top.iterrows():
        ax.annotate(row["factor_name"], (row["rank_ic_mean"], row["top_bottom_sharpe_naive"]), fontsize=8)
    ax.set_title("Factor quality: Rank IC mean vs top-bottom Sharpe")
    ax.set_xlabel("Rank IC mean")
    ax.set_ylabel("Top-bottom Sharpe")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "summary_rank_ic_vs_sharpe.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    panel_fp = args.panel_file if args.panel_file.is_absolute() else PROJECT_ROOT / args.panel_file
    output_dir = ensure_dir(args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir)
    per_factor_dir = ensure_dir(output_dir / "per_factor")

    panel_df = pd.read_parquet(panel_fp)
    builder = FactorBuilder()
    factor_names = builder.list_factors()
    factor_df = builder.compute_many(panel_df, factor_names)

    evaluator = FactorEvaluator(price_col=args.price_col)

    all_rows: list[dict] = []
    failures: list[dict[str, str]] = []

    factor_df.to_parquet(output_dir / "all_factor_values.parquet", index=False)

    for factor_name in factor_names:
        factor_out = ensure_dir(per_factor_dir / factor_name)
        try:
            result = evaluator.evaluate_one_factor(
                panel_df=panel_df,
                factor_df=factor_df,
                factor_name=factor_name,
                horizon=args.horizon,
                group_num=args.group_num,
            )
        except Exception as exc:
            failures.append({"factor_name": factor_name, "error": str(exc)})
            continue

        summary = result["summary"].copy()
        all_rows.append(summary)

        pd.DataFrame([summary]).to_csv(factor_out / "summary.csv", index=False)
        result["merged"].to_parquet(factor_out / "merged.parquet", index=False)
        save_series_csv(result["ic_series"], factor_out / "ic_series.csv", "ic")
        save_series_csv(result["rank_ic_series"], factor_out / "rank_ic_series.csv", "rank_ic")
        result["quantile_returns"].reset_index().to_csv(factor_out / "quantile_returns.csv", index=False)
        result["quantile_counts"].reset_index().to_csv(factor_out / "quantile_counts.csv", index=False)
        save_series_csv(result["top_bottom_returns"], factor_out / "top_bottom_returns.csv", "top_bottom_return")
        save_series_csv(result["top_bottom_cumret"], factor_out / "top_bottom_cumret.csv", "top_bottom_cumret")
        save_series_csv(result["factor_autocorr"], factor_out / "factor_autocorr.csv", "factor_autocorr")
        result["group_turnover"].to_csv(factor_out / "group_turnover.csv", index=False)
        save_series_csv(result["coverage_by_date"], factor_out / "coverage_by_date.csv", "coverage")
        result["distribution_stats"].reset_index().to_csv(factor_out / "distribution_stats.csv", index=False)
        result["yearly_ic_table"].to_csv(factor_out / "yearly_ic_table.csv", index=False)

        plot_full_report_to_file(result, factor_name, factor_out / "full_report.png")
        plot_quantile_bar_to_file(result, factor_name, factor_out / "quantile_bar.png")
        plot_yearly_ic_table_to_file(result, factor_name, factor_out / "yearly_ic_table.png")

    summary_df = pd.DataFrame(all_rows).sort_values("rank_ic_mean", ascending=False).reset_index(drop=True)
    summary_df.to_csv(output_dir / "single_factor_summary_all.csv", index=False)
    if failures:
        pd.DataFrame(failures).to_csv(output_dir / "single_factor_failures.csv", index=False)
    else:
        pd.DataFrame(columns=["factor_name", "error"]).to_csv(output_dir / "single_factor_failures.csv", index=False)

    metadata = {
        "panel_fp": str(panel_fp),
        "output_dir": str(output_dir),
        "horizon": int(args.horizon),
        "group_num": int(args.group_num),
        "price_col": args.price_col,
        "n_factors": len(factor_names),
        "factor_names": factor_names,
        "n_success": int(len(summary_df)),
        "n_failures": int(len(failures)),
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if not summary_df.empty:
        plot_global_summary(summary_df, output_dir)

    print(json.dumps(metadata, indent=2))
    print("saved summary to", output_dir / "single_factor_summary_all.csv")


if __name__ == "__main__":
    main()
