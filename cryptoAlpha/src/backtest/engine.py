from __future__ import annotations

import numpy as np
import pandas as pd

from src.factors.factor_utils import ensure_panel_sorted, forward_return


class FactorBacktestEngine:
	def __init__(self, price_col: str = "close") -> None:
		self.price_col = price_col

	def prepare_data(
		self,
		panel_df: pd.DataFrame,
		factor_df: pd.DataFrame,
		factor_name: str,
		horizon: int = 1,
	) -> pd.DataFrame:
		panel = ensure_panel_sorted(panel_df)
		factors = ensure_panel_sorted(factor_df)

		if factor_name not in factors.columns:
			raise ValueError(f"factor_name={factor_name} not in factor_df")

		ret_df = panel[["datetime", "symbol", self.price_col]].copy()
		ret_df["fwd_ret"] = forward_return(panel, price_col=self.price_col, horizon=horizon)

		merged = ret_df.merge(
			factors[["datetime", "symbol", factor_name]],
			on=["datetime", "symbol"],
			how="inner",
		)
		merged = merged.rename(columns={factor_name: "factor"})
		merged = merged.dropna(subset=["factor", "fwd_ret"]).reset_index(drop=True)
		return merged

	def make_top_bottom_equal_weight_positions(
		self,
		merged_df: pd.DataFrame,
		q: float = 0.2,
	) -> pd.DataFrame:
		if not (0 < q < 0.5):
			raise ValueError("q must be in (0, 0.5)")

		df = merged_df.copy()
		df["weight"] = 0.0

		def _assign(group: pd.DataFrame) -> pd.DataFrame:
			g = group.copy().sort_values("factor")
			n = len(g)
			if n < 4:
				return g

			k = max(1, int(np.floor(n * q)))
			short_idx = g.index[:k]
			long_idx = g.index[-k:]

			g.loc[short_idx, "weight"] = -1.0 / k
			g.loc[long_idx, "weight"] = 1.0 / k
			return g

		df = df.groupby("datetime", group_keys=False).apply(_assign)
		return df

	def make_zscore_weight_positions(
		self,
		merged_df: pd.DataFrame,
		max_abs_weight: float | None = 0.1,
	) -> pd.DataFrame:
		df = merged_df.copy()

		def _assign(group: pd.DataFrame) -> pd.DataFrame:
			g = group.copy()
			mean = g["factor"].mean()
			std = g["factor"].std()
			z = (g["factor"] - mean) / (std + 1e-12)

			if z.abs().sum() < 1e-12:
				g["weight"] = 0.0
				return g

			w = z / z.abs().sum()

			if max_abs_weight is not None:
				w = w.clip(-max_abs_weight, max_abs_weight)
				denom = w.abs().sum()
				if denom > 1e-12:
					w = w / denom

			g["weight"] = w
			return g

		df = df.groupby("datetime", group_keys=False).apply(_assign)
		return df

	def run_backtest(
		self,
		panel_df: pd.DataFrame,
		factor_df: pd.DataFrame,
		factor_name: str,
		horizon: int = 1,
		method: str = "top_bottom_equal_weight",
		quantile: float = 0.2,
		max_abs_weight: float | None = 0.1,
	) -> dict:
		merged = self.prepare_data(
			panel_df=panel_df,
			factor_df=factor_df,
			factor_name=factor_name,
			horizon=horizon,
		)

		if method == "top_bottom_equal_weight":
			pos = self.make_top_bottom_equal_weight_positions(merged, q=quantile)
		elif method == "zscore_weight":
			pos = self.make_zscore_weight_positions(merged, max_abs_weight=max_abs_weight)
		else:
			raise ValueError(f"Unsupported method: {method}")

		pos["pnl"] = pos["weight"] * pos["fwd_ret"]

		portfolio_ret = pos.groupby("datetime")["pnl"].sum().sort_index()
		gross_exposure = pos.groupby("datetime")["weight"].apply(lambda s: s.abs().sum()).sort_index()

		turnover = (
			pos.pivot(index="datetime", columns="symbol", values="weight")
			.fillna(0.0)
			.diff()
			.abs()
			.sum(axis=1)
		)

		out = {
			"positions": pos,
			"portfolio_returns": portfolio_ret,
			"cum_returns": portfolio_ret.cumsum(),
			"gross_exposure": gross_exposure,
			"turnover": turnover,
		}
		return out
