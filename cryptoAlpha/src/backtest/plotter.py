from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


class BacktestPlotter:
    def plot_basic_report(
        self,
        bt_result: dict,
        title: str = "Backtest Report",
        figsize: tuple[int, int] = (14, 12),
    ) -> None:
        fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)

        bt_result["cum_returns"].plot(ax=axes[0], title=f"{title} - Cumulative Return")
        axes[0].axhline(0, linestyle="--")

        dd = bt_result["cum_returns"] - bt_result["cum_returns"].cummax()
        dd.plot(ax=axes[1], title=f"{title} - Drawdown")
        axes[1].axhline(0, linestyle="--")

        bt_result["turnover"].plot(ax=axes[2], title=f"{title} - Turnover")

        bt_result["gross_exposure"].plot(ax=axes[3], title=f"{title} - Gross Exposure")

        plt.tight_layout()
        plt.show()