from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


class MultiHorizonPlotter:
    def plot_decay_report(
        self,
        mh_result: dict,
        figsize: tuple[int, int] = (14, 10),
    ) -> None:
        factor_name = mh_result["factor_name"]
        summary_df: pd.DataFrame = mh_result["summary_df"]

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()

        # 1. IC decay
        axes[0].plot(summary_df["horizon"], summary_df["ic_mean"], marker="o")
        axes[0].axhline(0, linestyle="--")
        axes[0].set_title(f"{factor_name} IC Decay")
        axes[0].set_xlabel("Horizon (hours)")
        axes[0].set_ylabel("IC Mean")

        # 2. RankIC decay
        axes[1].plot(summary_df["horizon"], summary_df["rank_ic_mean"], marker="o")
        axes[1].axhline(0, linestyle="--")
        axes[1].set_title(f"{factor_name} RankIC Decay")
        axes[1].set_xlabel("Horizon (hours)")
        axes[1].set_ylabel("RankIC Mean")

        # 3. Top-Bottom mean return by horizon
        axes[2].bar(summary_df["horizon"].astype(str), summary_df["top_bottom_mean_ret"])
        axes[2].axhline(0, linestyle="--")
        axes[2].set_title(f"{factor_name} Top-Bottom Mean Return by Horizon")
        axes[2].set_xlabel("Horizon (hours)")
        axes[2].set_ylabel("Mean Return")

        # 4. Top-Bottom sharpe by horizon
        axes[3].bar(summary_df["horizon"].astype(str), summary_df["top_bottom_sharpe_naive"])
        axes[3].axhline(0, linestyle="--")
        axes[3].set_title(f"{factor_name} Top-Bottom Sharpe by Horizon")
        axes[3].set_xlabel("Horizon (hours)")
        axes[3].set_ylabel("Sharpe")

        plt.tight_layout()
        plt.show()

    def plot_cumret_by_horizon(
        self,
        mh_result: dict,
        figsize: tuple[int, int] = (12, 6),
    ) -> None:
        factor_name = mh_result["factor_name"]
        result_by_horizon: dict[int, dict] = mh_result["result_by_horizon"]

        plt.figure(figsize=figsize)
        for h, result in result_by_horizon.items():
            result["top_bottom_cumret"].plot(label=f"{h}h")

        plt.axhline(0, linestyle="--")
        plt.title(f"{factor_name} Top-Bottom Cumulative Return by Horizon")
        plt.xlabel("datetime")
        plt.ylabel("Cumulative Return")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_ic_series_grid(
        self,
        mh_result: dict,
        figsize: tuple[int, int] = (14, 10),
    ) -> None:
        factor_name = mh_result["factor_name"]
        result_by_horizon: dict[int, dict] = mh_result["result_by_horizon"]
        horizons = list(result_by_horizon.keys())

        n = len(horizons)
        ncols = 2
        nrows = (n + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=False)
        axes = axes.flatten()

        for i, h in enumerate(horizons):
            ax = axes[i]
            result_by_horizon[h]["ic_series"].plot(ax=ax, title=f"{factor_name} IC Series ({h}h)")
            ax.axhline(0, linestyle="--")

        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        plt.show()