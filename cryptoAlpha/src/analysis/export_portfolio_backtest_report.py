from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
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


ANNUALIZATION_HOURS = 24 * 365


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export full portfolio backtest report from saved artifacts.")
    parser.add_argument(
        "--backtest-dir",
        type=Path,
        default=Path("data/backtest_4h"),
        help="Backtest artifact directory, relative to project root by default.",
    )
    parser.add_argument("--output-label", type=str, default="4h")
    parser.add_argument("--date-tag", type=str, default="20251218_20251228")
    return parser.parse_args()


def make_absolute(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_series_csv(path: Path, value_name: str) -> pd.Series:
    df = pd.read_csv(path)
    if value_name not in df.columns:
        value_col = [c for c in df.columns if c != "datetime"][0]
    else:
        value_col = value_name
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.set_index("datetime")[value_col].sort_index()


def summarize_portfolio(portfolio_return: pd.Series, cumret: pd.Series, turnover: pd.Series) -> dict[str, float]:
    ret = portfolio_return.dropna()
    nav = cumret.dropna()
    dd = nav / nav.cummax() - 1.0

    mean_ret = ret.mean()
    vol = ret.std()
    ann_ret = mean_ret * ANNUALIZATION_HOURS
    ann_vol = vol * np.sqrt(ANNUALIZATION_HOURS)
    sharpe = ann_ret / (ann_vol + 1e-12)

    downside = ret[ret < 0].std()
    sortino = ann_ret / (downside * np.sqrt(ANNUALIZATION_HOURS) + 1e-12)
    max_drawdown = float(dd.min()) if not dd.empty else np.nan
    calmar = ann_ret / (abs(max_drawdown) + 1e-12)

    return {
        "n_periods": int(ret.shape[0]),
        "mean_ret": float(mean_ret),
        "vol": float(vol),
        "annual_return": float(ann_ret),
        "annual_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "cum_return_last": float(nav.iloc[-1] - 1.0),
        "max_drawdown": float(max_drawdown),
        "calmar": float(calmar),
        "hit_rate": float((ret > 0).mean()),
        "avg_turnover": float(turnover.mean()),
    }


def yearly_table(portfolio_return: pd.Series) -> pd.DataFrame:
    ret = portfolio_return.dropna().to_frame("portfolio_return")
    index_ts = pd.to_datetime(pd.Series(ret.index, index=ret.index))
    ret["year"] = index_ts.dt.year.astype(int).to_numpy()
    rows: list[dict[str, float | int]] = []

    for year in sorted(ret["year"].dropna().unique().tolist()):
        year_int = int(year)
        group = ret[ret["year"] == year_int]
        g = group["portfolio_return"]
        cum = float(np.prod((1.0 + g).to_numpy(dtype=float))) - 1.0
        vol = float(g.std())
        ann_ret = float(g.mean()) * ANNUALIZATION_HOURS
        ann_vol = vol * float(np.sqrt(ANNUALIZATION_HOURS))
        sharpe = ann_ret / (ann_vol + 1e-12)
        rows.append(
            {
                "year": year_int,
                "n_periods": int(g.shape[0]),
                "return": cum,
                "mean_ret": float(g.mean()),
                "vol": vol,
                "annual_return": ann_ret,
                "annual_vol": ann_vol,
                "sharpe": float(sharpe),
                "hit_rate": float((g > 0).mean()),
            }
        )

    return pd.DataFrame(rows)


def plot_full_report(
    output_path: Path,
    title: str,
    portfolio_return: pd.Series,
    cumret: pd.Series,
    drawdown: pd.Series,
    turnover: pd.Series,
    long_count: pd.Series,
    short_count: pd.Series,
    gross_exposure: pd.Series,
    net_exposure: pd.Series,
) -> None:
    fig, axes = plt.subplots(6, 1, figsize=(16, 20), sharex=True)

    (cumret - 1.0).plot(ax=axes[0], color="#1f77b4", title=f"{title} - Cumulative Return")
    axes[0].axhline(0.0, linestyle="--", color="gray", linewidth=1)

    axes[1].fill_between(drawdown.index, drawdown.values, 0.0, color="#d62728", alpha=0.35)
    axes[1].set_title(f"{title} - Drawdown")

    portfolio_return.rolling(24).mean().plot(ax=axes[2], color="#2ca02c", title=f"{title} - 24h Rolling Mean Return")
    axes[2].axhline(0.0, linestyle="--", color="gray", linewidth=1)

    turnover.rolling(24).mean().plot(ax=axes[3], color="#9467bd", title=f"{title} - 24h Rolling Turnover")

    long_count.plot(ax=axes[4], label="long_count", color="#2ca02c")
    short_count.plot(ax=axes[4], label="short_count", color="#d62728")
    axes[4].set_title(f"{title} - Holding Counts")
    axes[4].legend()

    gross_exposure.plot(ax=axes[5], label="gross_exposure", color="black")
    net_exposure.plot(ax=axes[5], label="net_exposure", color="#ff7f0e")
    axes[5].set_title(f"{title} - Exposure")
    axes[5].legend()

    plt.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_nav_only(output_path: Path, title: str, cumret: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    (cumret - 1.0).plot(ax=ax, color="#1f77b4")
    ax.axhline(0.0, linestyle="--", color="gray", linewidth=1)
    ax.set_title(f"{title} - NAV")
    ax.set_ylabel("Cumulative Return")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_bar(output_path: Path, series: pd.Series, title: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    series.plot(kind="bar", ax=ax, color=["#2ca02c" if x >= 0 else "#d62728" for x in series.fillna(0.0)])
    ax.axhline(0.0, linestyle="--", color="gray", linewidth=1)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    backtest_dir = make_absolute(args.backtest_dir)
    label = args.output_label
    date_tag = args.date_tag

    holding_fp = backtest_dir / f"ridge_holding_df_{label}_{date_tag}.parquet"
    weight_fp = backtest_dir / f"ridge_weight_df_{label}_{date_tag}.parquet"
    ret_fp = backtest_dir / f"ridge_portfolio_return_{label}_{date_tag}.csv"
    cumret_fp = backtest_dir / f"ridge_cumret_{label}_{date_tag}.csv"
    summary_fp = backtest_dir / f"ridge_portfolio_summary_{label}_{date_tag}.csv"
    metadata_fp = backtest_dir / f"ridge_pipeline_metadata_{label}_{date_tag}.json"

    holding_df = pd.read_parquet(holding_fp)
    weight_df = pd.read_parquet(weight_fp)
    portfolio_return = read_series_csv(ret_fp, "portfolio_return")
    cumret = read_series_csv(cumret_fp, "cumret")

    holding_df["datetime"] = pd.to_datetime(holding_df["datetime"])
    weight_df["datetime"] = pd.to_datetime(weight_df["datetime"])

    weight_panel = (
        holding_df.pivot(index="datetime", columns="symbol", values="weight")
        .fillna(0.0)
        .sort_index()
    )
    turnover = weight_panel.diff().abs().sum(axis=1).rename("turnover")
    long_count = (weight_panel > 0).sum(axis=1).rename("long_count")
    short_count = (weight_panel < 0).sum(axis=1).rename("short_count")
    gross_exposure = weight_panel.abs().sum(axis=1).rename("gross_exposure")
    net_exposure = weight_panel.sum(axis=1).rename("net_exposure")
    drawdown = (cumret / cumret.cummax() - 1.0).rename("drawdown")

    monthly_return = ((1.0 + portfolio_return).resample("ME").prod() - 1.0).rename("monthly_return")
    yearly_return_series = ((1.0 + portfolio_return).resample("YE").prod() - 1.0).rename("yearly_return")
    yearly_stats = yearly_table(portfolio_return)

    extended_summary = summarize_portfolio(portfolio_return, cumret, turnover)
    if summary_fp.exists():
        base_summary = pd.read_csv(summary_fp)
    else:
        base_summary = pd.DataFrame()
    extended_summary_df = pd.DataFrame([extended_summary]).add_prefix("extended_")
    merged_summary_df = pd.concat([base_summary, extended_summary_df], axis=1)

    drawdown.reset_index().to_csv(backtest_dir / f"ridge_drawdown_{label}_{date_tag}.csv", index=False)
    turnover.reset_index().to_csv(backtest_dir / f"ridge_turnover_{label}_{date_tag}.csv", index=False)
    pd.concat([gross_exposure, net_exposure], axis=1).reset_index().to_csv(
        backtest_dir / f"ridge_exposure_{label}_{date_tag}.csv", index=False
    )
    pd.concat([long_count, short_count], axis=1).reset_index().to_csv(
        backtest_dir / f"ridge_holding_counts_{label}_{date_tag}.csv", index=False
    )
    monthly_return.reset_index().to_csv(backtest_dir / f"ridge_monthly_return_{label}_{date_tag}.csv", index=False)
    yearly_return_series.reset_index().to_csv(backtest_dir / f"ridge_yearly_return_{label}_{date_tag}.csv", index=False)
    yearly_stats.to_csv(backtest_dir / f"ridge_yearly_stats_{label}_{date_tag}.csv", index=False)
    merged_summary_df.to_csv(backtest_dir / f"ridge_portfolio_summary_full_{label}_{date_tag}.csv", index=False)

    title = f"Ridge Portfolio {label} {date_tag}"
    plot_full_report(
        backtest_dir / f"ridge_backtest_report_{label}_{date_tag}.png",
        title,
        portfolio_return,
        cumret,
        drawdown,
        turnover,
        long_count,
        short_count,
        gross_exposure,
        net_exposure,
    )
    plot_nav_only(backtest_dir / f"ridge_nav_{label}_{date_tag}.png", title, cumret)
    plot_bar(
        backtest_dir / f"ridge_monthly_return_{label}_{date_tag}.png",
        monthly_return,
        f"{title} - Monthly Returns",
        "Monthly Return",
    )
    plot_bar(
        backtest_dir / f"ridge_yearly_return_{label}_{date_tag}.png",
        yearly_return_series,
        f"{title} - Yearly Returns",
        "Yearly Return",
    )

    report_meta = {
        "backtest_dir": str(backtest_dir),
        "output_label": label,
        "date_tag": date_tag,
        "files": {
            "holding_fp": str(holding_fp),
            "weight_fp": str(weight_fp),
            "portfolio_return_fp": str(ret_fp),
            "cumret_fp": str(cumret_fp),
            "summary_fp": str(summary_fp),
            "metadata_fp": str(metadata_fp),
            "full_summary_fp": str(backtest_dir / f"ridge_portfolio_summary_full_{label}_{date_tag}.csv"),
            "report_png": str(backtest_dir / f"ridge_backtest_report_{label}_{date_tag}.png"),
            "nav_png": str(backtest_dir / f"ridge_nav_{label}_{date_tag}.png"),
        },
        "extended_summary": extended_summary,
    }
    (backtest_dir / f"ridge_backtest_report_metadata_{label}_{date_tag}.json").write_text(
        json.dumps(report_meta, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    print(json.dumps(report_meta, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
