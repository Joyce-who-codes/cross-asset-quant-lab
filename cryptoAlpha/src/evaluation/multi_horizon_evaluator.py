from __future__ import annotations

import pandas as pd

from src.evaluation.factor_evaluator import FactorEvaluator


class MultiHorizonEvaluator:
    def __init__(self, price_col: str = "close") -> None:
        self.price_col = price_col
        self.single_evaluator = FactorEvaluator(price_col=price_col)

    def evaluate_one_factor_multi_horizon(
        self,
        panel_df: pd.DataFrame,
        factor_df: pd.DataFrame,
        factor_name: str,
        horizons: list[int] | None = None,
        group_num: int = 5,
    ) -> dict:
        horizons = horizons or [1, 3, 6, 12, 24]

        result_by_horizon: dict[int, dict] = {}
        summary_rows: list[dict] = []

        for h in horizons:
            result = self.single_evaluator.evaluate_one_factor(
                panel_df=panel_df,
                factor_df=factor_df,
                factor_name=factor_name,
                horizon=h,
                group_num=group_num,
            )
            result_by_horizon[h] = result

            row = result["summary"].copy()
            row["horizon"] = h
            summary_rows.append(row)

        summary_df = (
            pd.DataFrame(summary_rows)
            .sort_values("horizon")
            .reset_index(drop=True)
        )

        return {
            "factor_name": factor_name,
            "horizons": horizons,
            "summary_df": summary_df,
            "result_by_horizon": result_by_horizon,
        }

    def evaluate_many_factors_multi_horizon(
        self,
        panel_df: pd.DataFrame,
        factor_df: pd.DataFrame,
        factor_names: list[str],
        horizons: list[int] | None = None,
        group_num: int = 5,
    ) -> pd.DataFrame:
        horizons = horizons or [1, 3, 6, 12, 24]

        rows: list[dict] = []

        for factor_name in factor_names:
            for h in horizons:
                result = self.single_evaluator.evaluate_one_factor(
                    panel_df=panel_df,
                    factor_df=factor_df,
                    factor_name=factor_name,
                    horizon=h,
                    group_num=group_num,
                )
                row = result["summary"].copy()
                row["factor_name"] = factor_name
                row["horizon"] = h
                rows.append(row)

        return (
            pd.DataFrame(rows)
            .sort_values(["factor_name", "horizon"])
            .reset_index(drop=True)
        )