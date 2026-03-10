from __future__ import annotations

import numpy as np
import pandas as pd


def calc_drawdown(cum_returns: pd.Series) -> pd.Series:
	running_max = cum_returns.cummax()
	return cum_returns - running_max


def summarize_backtest(
	portfolio_returns: pd.Series,
	turnover: pd.Series | None = None,
	periods_per_year: int = 24 * 365,
) -> dict:
	ret = portfolio_returns.dropna()
	if ret.empty:
		return {}

	cumret = ret.cumsum()
	dd = calc_drawdown(cumret)

	ann_ret = ret.mean() * periods_per_year
	ann_vol = ret.std() * np.sqrt(periods_per_year)
	sharpe = ann_ret / (ann_vol + 1e-12)

	summary = {
		"n_periods": int(ret.shape[0]),
		"mean_ret": float(ret.mean()),
		"std_ret": float(ret.std()),
		"annual_return": float(ann_ret),
		"annual_vol": float(ann_vol),
		"sharpe": float(sharpe),
		"max_drawdown": float(dd.min()),
		"hit_rate": float((ret > 0).mean()),
	}

	if turnover is not None and not turnover.dropna().empty:
		summary["avg_turnover"] = float(turnover.mean())

	return summary
