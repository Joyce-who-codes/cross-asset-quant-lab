from __future__ import annotations

import matplotlib.pyplot as plt


class PortfolioPlotter:

    def plot_basic_report(
        self,
        bt_result: dict,
        title: str = "Portfolio Report",
        figsize: tuple[int, int] = (14, 14),
        turnover_smooth: int = 24,
    ) -> None:

        cumret = bt_result["cumret"]
        turnover = bt_result["turnover"]

        # convert to cumulative return %
        cumret_pct = cumret - 1.0

        dd = cumret / cumret.cummax() - 1.0

        if turnover_smooth > 1:
            turnover = turnover.rolling(turnover_smooth).mean()

        fig, axes = plt.subplots(5, 1, figsize=figsize, sharex=True)

        # cumulative return
        cumret_pct.plot(
            ax=axes[0],
            title=f"{title} - Cumulative Return",
            color="navy",
        )
        axes[0].axhline(0.0, linestyle="--", color="gray")

        # drawdown
        axes[1].fill_between(
            dd.index,
            dd.values,
            0,
            color="red",
            alpha=0.3,
        )
        axes[1].set_title(f"{title} - Drawdown")

        # turnover
        turnover.plot(
            ax=axes[2],
            title=f"{title} - Turnover",
            color="purple",
        )

        # holding counts
        bt_result["long_count"].plot(
            ax=axes[3],
            label="long_count",
            color="green",
        )
        bt_result["short_count"].plot(
            ax=axes[3],
            label="short_count",
            color="red",
        )
        axes[3].set_title(f"{title} - Holding Counts")
        axes[3].legend()

        # exposure
        bt_result["gross_exposure"].plot(
            ax=axes[4],
            label="gross_exposure",
            color="black",
        )
        bt_result["net_exposure"].plot(
            ax=axes[4],
            label="net_exposure",
            color="orange",
        )
        axes[4].set_title(f"{title} - Exposure")
        axes[4].legend()

        plt.tight_layout()
        plt.show()