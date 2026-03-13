from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


class FactorPlotter:
    def plot_full_report(
        self,
        result: dict,
        factor_name: str,
        figsize: tuple[int, int] = (16, 18),
    ) -> None:
        fig, axes = plt.subplots(4, 2, figsize=figsize)
        axes = axes.flatten()

        # 1. quantile cumulative returns
        qcum = result["quantile_returns"].cumsum()
        qcum.plot(ax=axes[0], title=f"{factor_name} Quantile Cumulative Returns")
        axes[0].axhline(0, linestyle="--")

        # 2. top-bottom cumulative returns
        result["top_bottom_cumret"].plot(ax=axes[1], title=f"{factor_name} Top-Bottom Cumulative Return")
        axes[1].axhline(0, linestyle="--")

        # 3. IC series
        result["ic_series"].plot(ax=axes[2], title=f"{factor_name} IC Series")
        axes[2].axhline(0, linestyle="--")

        # 4. RankIC series
        result["rank_ic_series"].plot(ax=axes[3], title=f"{factor_name} RankIC Series")
        axes[3].axhline(0, linestyle="--")

        # 5. IC histogram
        result["ic_series"].dropna().hist(ax=axes[4], bins=40)
        axes[4].set_title(f"{factor_name} IC Histogram")

        # 6. factor autocorr
        result["factor_autocorr"].plot(ax=axes[5], title=f"{factor_name} Factor Autocorrelation")
        axes[5].axhline(0, linestyle="--")

        # 7. coverage
        result["coverage_by_date"].plot(ax=axes[6], title=f"{factor_name} Coverage by Date")
        axes[6].axhline(result["coverage_by_date"].mean(), linestyle="--")

        # 8. group turnover
        turnover_df = result["group_turnover"]
        if not turnover_df.empty:
            pivot_to = turnover_df.pivot(index="datetime", columns="quantile", values="turnover")
            pivot_to.plot(ax=axes[7], title=f"{factor_name} Quantile Turnover")
        else:
            axes[7].set_title(f"{factor_name} Quantile Turnover (No Data)")

        plt.tight_layout()
        plt.show()

    def plot_quantile_bar(
        self,
        result: dict,
        factor_name: str,
        figsize: tuple[int, int] = (10, 4),
    ) -> None:
        mean_ret = result["quantile_returns"].mean()
        plt.figure(figsize=figsize)
        mean_ret.plot(kind="bar")
        plt.title(f"{factor_name} Mean Return by Quantile")
        plt.ylabel("Mean Forward Return")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_yearly_ic_table(self, result: dict, factor_name: str) -> pd.DataFrame:
        yearly = result["yearly_ic_table"].copy()
        print(f"=== {factor_name} Yearly IC Table ===")
        print(yearly)
        return yearly