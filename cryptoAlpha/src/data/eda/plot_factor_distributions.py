from __future__ import annotations

import argparse
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

from src.factors.factor_builder import FactorBuilder


DEFAULT_PANEL_FP = PROJECT_ROOT / "data" / "cache" / "panel_top20_1h_2025_20260311.parquet"
DEFAULT_STATS_FP = PROJECT_ROOT / "data" / "eda" / "low_freq_report" / "default_factor_descriptive_stats.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "eda" / "low_freq_report"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot low-frequency factor value distributions.")
    parser.add_argument("--panel-file", type=Path, default=DEFAULT_PANEL_FP)
    parser.add_argument("--stats-file", type=Path, default=DEFAULT_STATS_FP)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--clip-quantile", type=float, default=0.01)
    return parser.parse_args()


def make_absolute(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_factor_values(panel_fp: Path, factor_names: list[str]) -> pd.DataFrame:
    panel_df = pd.read_parquet(panel_fp)
    builder = FactorBuilder()
    factor_df = builder.compute_many(panel_df, factor_names)
    return factor_df[["datetime", "symbol", *factor_names]]


def plot_single_factor_distribution(
    series: pd.Series,
    factor_name: str,
    output_path: Path,
    clip_q: float,
) -> dict[str, float]:
    s = series.dropna().astype(float)
    lower = float(s.quantile(clip_q))
    upper = float(s.quantile(1.0 - clip_q))
    clipped = s.clip(lower=lower, upper=upper)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(s, bins=80, color="#4C78A8", alpha=0.85)
    axes[0].axvline(s.mean(), color="#E45756", linestyle="--", linewidth=1.5, label="mean")
    axes[0].axvline(s.median(), color="#54A24B", linestyle=":", linewidth=1.5, label="median")
    axes[0].set_title(f"{factor_name} - Raw distribution")
    axes[0].set_xlabel("factor value")
    axes[0].set_ylabel("count")
    axes[0].legend()
    axes[0].grid(alpha=0.2)

    axes[1].hist(clipped, bins=80, color="#F58518", alpha=0.85)
    axes[1].axvline(clipped.mean(), color="#E45756", linestyle="--", linewidth=1.5, label="mean")
    axes[1].axvline(clipped.median(), color="#54A24B", linestyle=":", linewidth=1.5, label="median")
    axes[1].set_title(f"{factor_name} - Clipped [{clip_q:.0%}, {1.0-clip_q:.0%}]")
    axes[1].set_xlabel("factor value")
    axes[1].set_ylabel("count")
    axes[1].legend()
    axes[1].grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    return {
        "factor_name": factor_name,
        "count": int(s.shape[0]),
        "mean": float(s.mean()),
        "std": float(s.std()),
        "min": float(s.min()),
        "p01": float(s.quantile(0.01)),
        "p05": float(s.quantile(0.05)),
        "median": float(s.median()),
        "p95": float(s.quantile(0.95)),
        "p99": float(s.quantile(0.99)),
        "max": float(s.max()),
        "clip_lower": lower,
        "clip_upper": upper,
    }


def plot_grid(factor_df: pd.DataFrame, factor_names: list[str], output_path: Path, clip_q: float) -> None:
    n = len(factor_names)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.6 * nrows))
    axes = pd.Series(axes.flatten())

    for ax, factor_name in zip(axes, factor_names):
        s = factor_df[factor_name].dropna().astype(float)
        lower = float(s.quantile(clip_q))
        upper = float(s.quantile(1.0 - clip_q))
        clipped = s.clip(lower=lower, upper=upper)
        ax.hist(clipped, bins=60, color="#4C78A8", alpha=0.9)
        ax.axvline(clipped.median(), color="#E45756", linestyle="--", linewidth=1.2)
        ax.set_title(factor_name)
        ax.grid(alpha=0.2)

    for ax in axes[len(factor_names):]:
        ax.axis("off")

    fig.suptitle(f"Default factor distributions (clipped at [{clip_q:.0%}, {1.0-clip_q:.0%}])", y=0.995)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_boxplot(factor_df: pd.DataFrame, factor_names: list[str], output_path: Path, clip_q: float) -> None:
    clipped_data: list[pd.Series] = []
    for factor_name in factor_names:
        s = factor_df[factor_name].dropna().astype(float)
        lower = float(s.quantile(clip_q))
        upper = float(s.quantile(1.0 - clip_q))
        clipped_data.append(s.clip(lower=lower, upper=upper))

    fig, ax = plt.subplots(figsize=(14, max(5, 0.5 * len(factor_names))))
    ax.boxplot(clipped_data, labels=factor_names, vert=False, showfliers=False)
    ax.set_title(f"Default factor boxplots (clipped at [{clip_q:.0%}, {1.0-clip_q:.0%}])")
    ax.set_xlabel("factor value")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    panel_fp = make_absolute(args.panel_file)
    stats_fp = make_absolute(args.stats_file)
    output_dir = ensure_dir(make_absolute(args.output_dir))
    per_factor_dir = ensure_dir(output_dir / "factor_distributions")

    stats_df = pd.read_csv(stats_fp)
    factor_names = stats_df["column"].dropna().astype(str).tolist()

    factor_df = load_factor_values(panel_fp, factor_names)

    rows: list[dict[str, float]] = []
    for factor_name in factor_names:
        row = plot_single_factor_distribution(
            factor_df[factor_name],
            factor_name,
            per_factor_dir / f"{factor_name}_distribution.png",
            clip_q=args.clip_quantile,
        )
        rows.append(row)

    plot_grid(
        factor_df,
        factor_names,
        output_dir / "default_factor_distribution_grid.png",
        clip_q=args.clip_quantile,
    )
    plot_boxplot(
        factor_df,
        factor_names,
        output_dir / "default_factor_boxplot_clipped.png",
        clip_q=args.clip_quantile,
    )

    pd.DataFrame(rows).to_csv(output_dir / "default_factor_distribution_summary.csv", index=False)
    print(output_dir / "default_factor_distribution_grid.png")


if __name__ == "__main__":
    main()
