from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def resolve_project_root(start: Path) -> Path:
    current = start.resolve()
    if current.is_file():
        current = current.parent
    while not (current / "src").exists():
        if current.parent == current:
            raise RuntimeError("Cannot find project root")
        current = current.parent
    return current


PROJECT_ROOT = resolve_project_root(Path(__file__).resolve())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.portfolio_ridge_pipeline import DEFAULT_FACTOR_NAMES
from src.data.registry import DataRegistry
from src.factors.factor_builder import FactorBuilder


PANEL_PATH = PROJECT_ROOT / "data" / "cache" / "panel_top20_1h_2025_20260311.parquet"
OUTPUT_DIR = PROJECT_ROOT / "data" / "eda" / "low_freq_report"


SOURCE_GROUPS: dict[str, dict[str, object]] = {
    "coinglass_price": {
        "source": "Coinglass",
        "dataset": "FUTURES_PRICE_HISTORY",
        "role": "Perpetual futures OHLCV price panel",
        "columns": ["open", "high", "low", "close", "volume_usd"],
        "root": "/data/qr/ALTERNATIVE_DATA/COINGLASS/COINGLASS_v6/NORMAL/FUTURES_PRICE_HISTORY/1h/Binance",
    },
    "coinglass_funding": {
        "source": "Coinglass",
        "dataset": "FUTURES_FUNDING_RATE_HISTORY",
        "role": "Funding-rate history for derivatives sentiment/carry",
        "columns": ["funding_open", "funding_high", "funding_low", "funding_close"],
        "root": "/data/qr/ALTERNATIVE_DATA/COINGLASS/COINGLASS_v6/NORMAL/FUTURES_FUNDING_RATE_HISTORY/1h/Binance",
    },
    "coinglass_oi": {
        "source": "Coinglass",
        "dataset": "FUTURES_OPEN_INTEREST",
        "role": "Open interest for leverage and positioning",
        "columns": ["oi_open", "oi_high", "oi_low", "oi_close"],
        "root": "/data/qr/ALTERNATIVE_DATA/COINGLASS/COINGLASS_v6/NORMAL/FUTURES_OPEN_INTEREST/1h/Binance",
    },
    "coinglass_taker": {
        "source": "Coinglass",
        "dataset": "FUTURES_TAKER_BUY_SELL_VOLUME",
        "role": "Aggressive buy/sell flow",
        "columns": ["taker_buy_volume_usd", "taker_sell_volume_usd"],
        "root": "/data/qr/ALTERNATIVE_DATA/COINGLASS/COINGLASS_v6/NORMAL/FUTURES_TAKER_BUY_SELL_VOLUME/1h/Binance",
    },
    "coinglass_ls": {
        "source": "Coinglass",
        "dataset": "FUTURES_GLOBAL_LS_ACCOUNT_RATIO",
        "role": "Global long-short account ratio",
        "columns": [
            "global_account_long_percent",
            "global_account_short_percent",
            "global_account_long_short_ratio",
        ],
        "root": "/data/qr/ALTERNATIVE_DATA/COINGLASS/COINGLASS_v6/NORMAL/FUTURES_GLOBAL_LS_ACCOUNT_RATIO/1h/Binance",
    },
    "coingecko_market": {
        "source": "CoinGecko",
        "dataset": "COIN_MARKET_CHART/hourly",
        "role": "Spot market price, market cap, total volume",
        "columns": ["price", "market_cap", "total_volume"],
        "root": "/data/qr/ALTERNATIVE_DATA/COINGECKO_v3/NORMAL/COIN_MARKET_CHART/hourly",
    },
    "cryptoracle_social": {
        "source": "Cryptoracle",
        "dataset": "selected social/sentiment metrics",
        "role": "Social activity and sentiment indicators",
        "columns": [
            "active_community_count",
            "mention_count",
            "positive_sentiment_ratio",
            "negative_sentiment_ratio",
        ],
        "root": "/home/joyce/cryptoracle/cryptoracle_data/NORMAL",
    },
}


FACTOR_DEPENDENCIES = {
    "mom_24h": ("Coinglass", ["close"]),
    "mom_6h": ("Coinglass", ["close"]),
    "funding_z_24": ("Coinglass", ["funding_close"]),
    "oi_change_24h": ("Coinglass", ["oi_close"]),
    "taker_imbalance": ("Coinglass", ["taker_buy_volume_usd", "taker_sell_volume_usd"]),
    "long_short_ratio_z_24": ("Coinglass", ["global_account_long_short_ratio"]),
    "volume_ratio_24": ("Coinglass", ["volume_usd"]),
    "active_community_count_z_24": ("Cryptoracle", ["active_community_count"]),
}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".csv":
        df.to_csv(path, index=False)
    elif path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        raise ValueError(f"Unsupported table extension: {path}")


def load_panel(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values(["datetime", "symbol"]).reset_index(drop=True)
    return df


def build_source_summary(panel_df: pd.DataFrame, registry: DataRegistry) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    factor_map: dict[str, list[str]] = {}
    for factor_name, (source_name, dep_cols) in FACTOR_DEPENDENCIES.items():
        for spec in SOURCE_GROUPS.values():
            spec_source = str(spec["source"])
            spec_columns = cast(list[str], spec["columns"])
            spec_dataset = str(spec["dataset"])
            if spec_source == source_name and any(col in spec_columns for col in dep_cols):
                key = spec_dataset
                factor_map.setdefault(key, []).append(factor_name)

    for spec in SOURCE_GROUPS.values():
        spec_source = str(spec["source"])
        spec_dataset = str(spec["dataset"])
        spec_root = str(spec["root"])
        spec_role = str(spec["role"])
        spec_columns = cast(list[str], spec["columns"])
        cols = [c for c in spec_columns if c in panel_df.columns]
        if not cols:
            continue
        non_null_ratio_by_col = {c: float(panel_df[c].notna().mean()) for c in cols}
        rows.append(
            {
                "source": spec_source,
                "dataset": spec_dataset,
                "raw_root": spec_root,
                "role": spec_role,
                "panel_columns": " | ".join(cols),
                "n_panel_columns": len(cols),
                "avg_non_null_ratio": float(np.mean(list(non_null_ratio_by_col.values()))),
                "min_non_null_ratio": float(np.min(list(non_null_ratio_by_col.values()))),
                "max_non_null_ratio": float(np.max(list(non_null_ratio_by_col.values()))),
                "used_model_factors": " | ".join(sorted(factor_map.get(spec_dataset, []))),
            }
        )
    return pd.DataFrame(rows).sort_values(["source", "dataset"]).reset_index(drop=True)


def build_factor_mapping_table() -> pd.DataFrame:
    rows = []
    for factor_name in DEFAULT_FACTOR_NAMES:
        source_name, dep_cols = FACTOR_DEPENDENCIES.get(factor_name, ("Unknown", []))
        rows.append(
            {
                "factor_name": factor_name,
                "source": source_name,
                "dependency_columns": " | ".join(dep_cols),
            }
        )
    return pd.DataFrame(rows)


def build_panel_overview(panel_df: pd.DataFrame) -> dict[str, object]:
    dt = panel_df["datetime"].sort_values().drop_duplicates()
    diffs = dt.diff().dropna()
    inferred_freq = str(diffs.mode().iloc[0]) if not diffs.empty else "NA"
    obs_per_time = panel_df.groupby("datetime")["symbol"].nunique()
    return {
        "n_rows": int(len(panel_df)),
        "n_columns": int(panel_df.shape[1]),
        "n_symbols": int(panel_df["symbol"].nunique()),
        "symbols": sorted(panel_df["symbol"].astype(str).unique().tolist()),
        "start_datetime": str(panel_df["datetime"].min()),
        "end_datetime": str(panel_df["datetime"].max()),
        "n_unique_datetimes": int(panel_df["datetime"].nunique()),
        "inferred_frequency": inferred_freq,
        "avg_symbols_per_timestamp": float(obs_per_time.mean()),
        "min_symbols_per_timestamp": int(obs_per_time.min()),
        "max_symbols_per_timestamp": int(obs_per_time.max()),
    }


def build_symbol_summary(panel_df: pd.DataFrame) -> pd.DataFrame:
    all_timestamps = int(panel_df["datetime"].nunique())
    rows = []
    for symbol, g in panel_df.groupby("symbol"):
        rows.append(
            {
                "symbol": symbol,
                "n_rows": int(len(g)),
                "start_datetime": str(g["datetime"].min()),
                "end_datetime": str(g["datetime"].max()),
                "coverage_ratio_vs_global_timestamps": float(len(g) / max(all_timestamps, 1)),
                "close_non_null_ratio": float(g["close"].notna().mean()) if "close" in g.columns else np.nan,
                "funding_non_null_ratio": float(g["funding_close"].notna().mean()) if "funding_close" in g.columns else np.nan,
                "oi_non_null_ratio": float(g["oi_close"].notna().mean()) if "oi_close" in g.columns else np.nan,
                "social_non_null_ratio": float(g["active_community_count"].notna().mean()) if "active_community_count" in g.columns else np.nan,
                "market_cap_non_null_ratio": float(g["market_cap"].notna().mean()) if "market_cap" in g.columns else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["n_rows", "symbol"], ascending=[False, True]).reset_index(drop=True)


def build_time_distribution(panel_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ts_counts = panel_df.groupby("datetime")["symbol"].nunique().reset_index(name="n_symbols")
    ts_counts["month"] = ts_counts["datetime"].dt.to_period("M").astype(str)
    monthly = ts_counts.groupby("month").agg(
        n_timestamps=("datetime", "count"),
        avg_symbols=("n_symbols", "mean"),
        min_symbols=("n_symbols", "min"),
        max_symbols=("n_symbols", "max"),
    ).reset_index()
    return ts_counts, monthly


def build_symbol_month_coverage(panel_df: pd.DataFrame) -> pd.DataFrame:
    df = panel_df[["datetime", "symbol"]].copy()
    df["month"] = df["datetime"].dt.to_period("M").astype(str)
    pivot = df.groupby(["symbol", "month"]).size().unstack(fill_value=0)
    return pivot.reset_index()


def build_numeric_stats(panel_df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for col in columns:
        if col not in panel_df.columns:
            continue
        s = pd.to_numeric(panel_df[col], errors="coerce")
        valid = s.dropna()
        if valid.empty:
            continue
        rows.append(
            {
                "column": col,
                "count": int(valid.shape[0]),
                "missing_ratio": float(s.isna().mean()),
                "mean": float(valid.mean()),
                "std": float(valid.std()),
                "min": float(valid.min()),
                "p5": float(valid.quantile(0.05)),
                "p25": float(valid.quantile(0.25)),
                "median": float(valid.median()),
                "p75": float(valid.quantile(0.75)),
                "p95": float(valid.quantile(0.95)),
                "max": float(valid.max()),
            }
        )
    return pd.DataFrame(rows)


def build_factor_eda(panel_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    factor_df = FactorBuilder().compute_many(panel_df, DEFAULT_FACTOR_NAMES)
    factor_stats = build_numeric_stats(factor_df, DEFAULT_FACTOR_NAMES)
    corr = factor_df[DEFAULT_FACTOR_NAMES].corr(numeric_only=True)
    return factor_stats, corr


def plot_symbol_counts(symbol_summary: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(symbol_summary["symbol"], symbol_summary["n_rows"], color="#4C78A8")
    ax.set_title("Observation count by symbol")
    ax.set_ylabel("Number of rows")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_monthly_coverage(monthly_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(monthly_df["month"], monthly_df["n_timestamps"], marker="o", color="#4C78A8", label="Hourly timestamps")
    ax1.set_ylabel("Number of timestamps", color="#4C78A8")
    ax1.tick_params(axis="y", labelcolor="#4C78A8")
    ax1.tick_params(axis="x", rotation=45)

    ax2 = ax1.twinx()
    ax2.plot(monthly_df["month"], monthly_df["avg_symbols"], marker="s", color="#F58518", label="Avg symbols per timestamp")
    ax2.set_ylabel("Average active symbols", color="#F58518")
    ax2.tick_params(axis="y", labelcolor="#F58518")

    ax1.set_title("Monthly time coverage of the low-frequency panel")
    ax1.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_source_non_null(source_summary: pd.DataFrame, output_path: Path) -> None:
    labels = source_summary["dataset"].astype(str)
    values = source_summary["avg_non_null_ratio"].astype(float)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(labels, values, color="#54A24B")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Average non-null ratio")
    ax.set_title("Column coverage by data source / dataset")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_symbol_month_heatmap(symbol_month_df: pd.DataFrame, output_path: Path) -> None:
    months = [c for c in symbol_month_df.columns if c != "symbol"]
    values = symbol_month_df[months].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(12, 7))
    im = ax.imshow(values, aspect="auto", cmap="YlGnBu")
    ax.set_yticks(range(len(symbol_month_df)))
    ax.set_yticklabels(symbol_month_df["symbol"])
    ax.set_xticks(range(len(months)))
    ax.set_xticklabels(months, rotation=45, ha="right")
    ax.set_title("Monthly observation density by symbol")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Number of rows")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_factor_corr(corr_df: pd.DataFrame, output_path: Path) -> None:
    vals = corr_df.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(vals, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(corr_df.columns)))
    ax.set_xticklabels(corr_df.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr_df.index)))
    ax.set_yticklabels(corr_df.index)
    ax.set_title("Correlation of default Ridge factors")
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            ax.text(j, i, f"{vals[i, j]:.2f}", ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Correlation")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_markdown_report(
    output_path: Path,
    overview: dict[str, object],
    source_summary: pd.DataFrame,
    symbol_summary: pd.DataFrame,
) -> None:
    top_symbols = symbol_summary.head(10)
    header = [str(c) for c in top_symbols.columns]
    rows = [header, ["---"] * len(header)]
    for row in top_symbols.itertuples(index=False):
        rows.append([str(v) for v in row])
    markdown_table = "\n".join(["| " + " | ".join(r) + " |" for r in rows])
    lines = [
        "# Low-frequency data summary for cryptoAlpha",
        "",
        "## Panel overview",
        f"- Rows: {overview['n_rows']:,}",
        f"- Columns: {overview['n_columns']}",
        f"- Symbols: {overview['n_symbols']}",
        f"- Start: {overview['start_datetime']}",
        f"- End: {overview['end_datetime']}",
        f"- Unique timestamps: {overview['n_unique_datetimes']:,}",
        f"- Inferred frequency: {overview['inferred_frequency']}",
        f"- Avg symbols per timestamp: {overview['avg_symbols_per_timestamp']:.2f}",
        "",
        "## Sources used in the low-frequency panel",
    ]
    for row in source_summary.to_dict(orient="records"):
        lines.extend(
            [
                f"- **{row['source']} / {row['dataset']}**: {row['role']}",
                f"  - Columns: {row['panel_columns']}",
                f"  - Avg non-null ratio: {row['avg_non_null_ratio']:.3f}",
                f"  - Model factors: {row['used_model_factors'] or 'not used directly in default Ridge factors'}",
            ]
        )
    lines.extend([
        "",
        "## Symbol coverage (top 10 by row count)",
        "",
        markdown_table,
    ])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    output_dir = ensure_dir(OUTPUT_DIR)
    panel_df = load_panel(PANEL_PATH)
    registry = DataRegistry()

    overview = build_panel_overview(panel_df)
    source_summary = build_source_summary(panel_df, registry)
    factor_mapping = build_factor_mapping_table()
    symbol_summary = build_symbol_summary(panel_df)
    timestamp_counts, monthly_summary = build_time_distribution(panel_df)
    symbol_month = build_symbol_month_coverage(panel_df)

    raw_numeric_cols = [
        "open", "high", "low", "close", "volume_usd",
        "funding_close", "oi_close", "taker_buy_volume_usd", "taker_sell_volume_usd",
        "global_account_long_short_ratio", "price", "market_cap", "total_volume",
        "active_community_count", "mention_count", "positive_sentiment_ratio", "negative_sentiment_ratio",
    ]
    raw_stats = build_numeric_stats(panel_df, raw_numeric_cols)
    factor_stats, factor_corr = build_factor_eda(panel_df)

    save_table(source_summary, output_dir / "source_summary.csv")
    save_table(factor_mapping, output_dir / "factor_source_mapping.csv")
    save_table(symbol_summary, output_dir / "symbol_summary.csv")
    save_table(timestamp_counts, output_dir / "timestamp_symbol_counts.csv")
    save_table(monthly_summary, output_dir / "monthly_time_distribution.csv")
    save_table(symbol_month, output_dir / "symbol_month_distribution.csv")
    save_table(raw_stats, output_dir / "raw_feature_descriptive_stats.csv")
    save_table(factor_stats, output_dir / "default_factor_descriptive_stats.csv")
    save_table(factor_corr.reset_index().rename(columns={"index": "factor"}), output_dir / "default_factor_correlation.csv")

    (output_dir / "panel_overview.json").write_text(json.dumps(overview, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown_report(output_dir / "low_freq_data_summary.md", overview, source_summary, symbol_summary)

    plot_symbol_counts(symbol_summary, output_dir / "symbol_observation_counts.png")
    plot_monthly_coverage(monthly_summary, output_dir / "monthly_time_coverage.png")
    plot_source_non_null(source_summary, output_dir / "source_non_null_ratio.png")
    plot_symbol_month_heatmap(symbol_month, output_dir / "symbol_month_coverage_heatmap.png")
    plot_factor_corr(factor_corr, output_dir / "default_factor_correlation_heatmap.png")

    print(json.dumps({
        "output_dir": str(output_dir),
        "overview": overview,
        "generated_tables": [
            "source_summary.csv",
            "factor_source_mapping.csv",
            "symbol_summary.csv",
            "timestamp_symbol_counts.csv",
            "monthly_time_distribution.csv",
            "symbol_month_distribution.csv",
            "raw_feature_descriptive_stats.csv",
            "default_factor_descriptive_stats.csv",
            "default_factor_correlation.csv",
            "panel_overview.json",
            "low_freq_data_summary.md",
        ],
        "generated_figures": [
            "symbol_observation_counts.png",
            "monthly_time_coverage.png",
            "source_non_null_ratio.png",
            "symbol_month_coverage_heatmap.png",
            "default_factor_correlation_heatmap.png",
        ],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
