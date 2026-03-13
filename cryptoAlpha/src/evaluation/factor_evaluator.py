from __future__ import annotations

import numpy as np
import pandas as pd

from src.factors.factor_utils import ensure_panel_sorted, forward_return


def _rank_corr(x: pd.Series, y: pd.Series) -> float:
    if x.notna().sum() < 3 or y.notna().sum() < 3:
        return np.nan
    return x.rank().corr(y.rank())


def _safe_corr(x: pd.Series, y: pd.Series) -> float:
    if x.notna().sum() < 3 or y.notna().sum() < 3:
        return np.nan
    return x.corr(y)


def _quantile_buckets(s: pd.Series, q: int) -> pd.Series:
    valid = s.dropna()
    if valid.nunique() < q:
        return pd.Series(index=s.index, dtype="float64")
    try:
        return pd.qcut(s, q=q, labels=False, duplicates="drop") + 1
    except Exception:
        return pd.Series(index=s.index, dtype="float64")


class FactorEvaluator:
    def __init__(self, price_col: str = "close") -> None:
        self.price_col = price_col

    def prepare_factor_label(
        self,
        panel_df: pd.DataFrame,
        factor_df: pd.DataFrame,
        factor_name: str,
        horizon: int = 1,
    ) -> pd.DataFrame:
        panel = ensure_panel_sorted(panel_df)
        factors = ensure_panel_sorted(factor_df)

        if factor_name not in factors.columns:
            raise ValueError(f"{factor_name} not found in factor_df")

        label_df = panel[["datetime", "symbol", self.price_col]].copy()
        label_df[f"label_ret_{horizon}h"] = forward_return(panel, price_col=self.price_col, horizon=horizon)

        merged = label_df.merge(
            factors[["datetime", "symbol", factor_name]],
            on=["datetime", "symbol"],
            how="inner",
        )
        merged = merged.rename(columns={factor_name: "factor"})
        merged = merged.dropna(subset=["factor", f"label_ret_{horizon}h"]).reset_index(drop=True)
        return merged

    def calc_ic_series(self, merged_df: pd.DataFrame, horizon: int) -> tuple[pd.Series, pd.Series]:
        label_col = f"label_ret_{horizon}h"

        ic = merged_df.groupby("datetime").apply(lambda g: _safe_corr(g["factor"], g[label_col]))
        rank_ic = merged_df.groupby("datetime").apply(lambda g: _rank_corr(g["factor"], g[label_col]))

        ic.name = "ic"
        rank_ic.name = "rank_ic"
        return ic, rank_ic

    def calc_quantile_returns(
        self,
        merged_df: pd.DataFrame,
        horizon: int,
        group_num: int = 5,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        label_col = f"label_ret_{horizon}h"
        df = merged_df.copy()

        df["factor_quantile"] = df.groupby("datetime")["factor"].transform(
            lambda s: _quantile_buckets(s, q=group_num)
        )

        qret = (
            df.dropna(subset=["factor_quantile"])
            .groupby(["datetime", "factor_quantile"])[label_col]
            .mean()
            .unstack()
            .sort_index()
        )

        qcount = (
            df.dropna(subset=["factor_quantile"])
            .groupby(["datetime", "factor_quantile"])["symbol"]
            .count()
            .unstack()
            .sort_index()
        )

        return qret, qcount

    def calc_factor_autocorr(self, merged_df: pd.DataFrame) -> pd.Series:
        df = merged_df[["datetime", "symbol", "factor"]].copy()
        df = df.sort_values(["symbol", "datetime"]).reset_index(drop=True)
        df["factor_lag1"] = df.groupby("symbol")["factor"].shift(1)

        ac = df.groupby("datetime").apply(lambda g: _safe_corr(g["factor"], g["factor_lag1"]))
        ac.name = "factor_autocorr"
        return ac

    def calc_group_turnover(
        self,
        merged_df: pd.DataFrame,
        group_num: int = 5,
    ) -> pd.DataFrame:
        df = merged_df.copy()
        df["factor_quantile"] = df.groupby("datetime")["factor"].transform(
            lambda s: _quantile_buckets(s, q=group_num)
        )
        df = df.dropna(subset=["factor_quantile"]).copy()
        df["factor_quantile"] = df["factor_quantile"].astype(int)

        date_list = sorted(df["datetime"].dropna().unique())
        rows = []

        prev_members: dict[int, set[str]] = {}

        for dt in date_list:
            sub = df[df["datetime"] == dt]
            for q in sorted(sub["factor_quantile"].unique()):
                members = set(sub.loc[sub["factor_quantile"] == q, "symbol"].astype(str))
                if q not in prev_members:
                    turnover = np.nan
                else:
                    prev = prev_members[q]
                    union_n = max(len(prev | members), 1)
                    turnover = 1.0 - len(prev & members) / union_n

                rows.append(
                    {
                        "datetime": dt,
                        "quantile": q,
                        "turnover": turnover,
                        "count": len(members),
                    }
                )
                prev_members[q] = members

        return pd.DataFrame(rows)

    def calc_coverage_by_date(self, merged_df: pd.DataFrame) -> pd.Series:
        cov = merged_df.groupby("datetime")["factor"].apply(lambda s: s.notna().mean())
        cov.name = "coverage"
        return cov

    def calc_distribution_stats(self, merged_df: pd.DataFrame) -> pd.DataFrame:
        g = merged_df.groupby("datetime")["factor"]
        out = pd.DataFrame(
            {
                "mean": g.mean(),
                "std": g.std(),
                "skew": g.skew(),
                "median": g.median(),
                "p01": g.quantile(0.01),
                "p99": g.quantile(0.99),
            }
        )
        return out

    def calc_yearly_ic_table(self, ic: pd.Series, rank_ic: pd.Series) -> pd.DataFrame:
        df = pd.concat([ic, rank_ic], axis=1).dropna(how="all")
        df["year"] = pd.to_datetime(df.index).year
        out = df.groupby("year").agg(
            ic_mean=("ic", "mean"),
            ic_std=("ic", "std"),
            rank_ic_mean=("rank_ic", "mean"),
            rank_ic_std=("rank_ic", "std"),
        )
        out["ic_ir"] = out["ic_mean"] / (out["ic_std"] + 1e-12)
        out["rank_ic_ir"] = out["rank_ic_mean"] / (out["rank_ic_std"] + 1e-12)
        return out.reset_index()

    def evaluate_one_factor(
        self,
        panel_df: pd.DataFrame,
        factor_df: pd.DataFrame,
        factor_name: str,
        horizon: int = 1,
        group_num: int = 5,
    ) -> dict:
        merged = self.prepare_factor_label(
            panel_df=panel_df,
            factor_df=factor_df,
            factor_name=factor_name,
            horizon=horizon,
        )
        if merged.empty:
            raise ValueError(f"No valid merged data for factor={factor_name}")

        ic, rank_ic = self.calc_ic_series(merged, horizon=horizon)
        qret, qcount = self.calc_quantile_returns(merged, horizon=horizon, group_num=group_num)
        fac_ac = self.calc_factor_autocorr(merged)
        turnover_df = self.calc_group_turnover(merged, group_num=group_num)
        coverage = self.calc_coverage_by_date(merged)
        dist_stats = self.calc_distribution_stats(merged)
        yearly_ic = self.calc_yearly_ic_table(ic, rank_ic)

        top_q = qret.columns.max()
        bottom_q = qret.columns.min()
        ls_ret = qret[top_q] - qret[bottom_q]
        ls_cumret = ls_ret.cumsum()

        summary = {
            "factor_name": factor_name,
            "horizon": horizon,
            "n_obs": int(len(merged)),
            "coverage_mean": float(coverage.mean()),
            "ic_mean": float(ic.mean()),
            "ic_std": float(ic.std()),
            "ic_ir": float(ic.mean() / (ic.std() + 1e-12)),
            "rank_ic_mean": float(rank_ic.mean()),
            "rank_ic_std": float(rank_ic.std()),
            "rank_ic_ir": float(rank_ic.mean() / (rank_ic.std() + 1e-12)),
            "factor_autocorr_mean": float(fac_ac.mean()),
            "top_bottom_mean_ret": float(ls_ret.mean()),
            "top_bottom_std_ret": float(ls_ret.std()),
            "top_bottom_sharpe_naive": float(ls_ret.mean() / (ls_ret.std() + 1e-12)),
            "avg_group_turnover": float(turnover_df["turnover"].mean(skipna=True)),
        }

        return {
            "summary": summary,
            "merged": merged,
            "ic_series": ic,
            "rank_ic_series": rank_ic,
            "quantile_returns": qret,
            "quantile_counts": qcount,
            "top_bottom_returns": ls_ret,
            "top_bottom_cumret": ls_cumret,
            "factor_autocorr": fac_ac,
            "group_turnover": turnover_df,
            "coverage_by_date": coverage,
            "distribution_stats": dist_stats,
            "yearly_ic_table": yearly_ic,
        }

    def evaluate_many_factors(
        self,
        panel_df: pd.DataFrame,
        factor_df: pd.DataFrame,
        factor_names: list[str],
        horizon: int = 1,
        group_num: int = 5,
    ) -> pd.DataFrame:
        rows = []
        for factor_name in factor_names:
            result = self.evaluate_one_factor(
                panel_df=panel_df,
                factor_df=factor_df,
                factor_name=factor_name,
                horizon=horizon,
                group_num=group_num,
            )
            rows.append(result["summary"])

        return pd.DataFrame(rows).sort_values("rank_ic_mean", ascending=False).reset_index(drop=True)