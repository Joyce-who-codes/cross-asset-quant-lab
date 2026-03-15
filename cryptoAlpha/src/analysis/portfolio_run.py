from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.factors.factor_builder import FactorBuilder
from src.models.xgb_alpha_model import XGBAlphaModel
from src.portfolio.portfolio_backtest import (
    backtest_long_short_portfolio,
    summarize_portfolio_result,
)
from src.portfolio.portfolio_plotter import PortfolioPlotter


FACTOR_NAMES = [
    "mom_24h",
    "mom_6h",
    "funding_z_24",
    "oi_change_24h",
    "taker_imbalance",
    "long_short_ratio_z_24",
    "volume_ratio_24",
    "active_community_count_z_24",
]


def main() -> None:
    print("PROJECT_ROOT =", PROJECT_ROOT)

    cache_dir = PROJECT_ROOT / "data" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    panel_fp = cache_dir / "panel_top20_1h_2025_20260311.parquet"
    panel_df = pd.read_parquet(panel_fp)

    builder = FactorBuilder()
    factor_df = builder.compute_many(panel_df, FACTOR_NAMES)

    print(factor_df.head())
    print(factor_df.shape)

    model = XGBAlphaModel(
        horizon=24,
        train_window=180,
    )

    df_model = model.build_label(panel_df)
    df_model = df_model.merge(
        factor_df,
        on=["datetime", "symbol"],
        how="left",
    )

    print(df_model.shape)
    print(df_model.head())

    pred_dir = PROJECT_ROOT / "data" / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    model_dir = PROJECT_ROOT / "data" / "models" / "daily_xgb_alpha"
    model_dir.mkdir(parents=True, exist_ok=True)

    pred_df = model.fit_predict_save_daily_models(
        df=df_model,
        feature_cols=FACTOR_NAMES,
        label_col="future_return",
        daily_model_dir=model_dir,
    )

    print(pred_df.shape)
    print(pred_df.head())

    pred_fp = pred_dir / "pred_df_top20_1h_2025_20260311.parquet"
    pred_df.to_parquet(pred_fp, index=False)

    print("saved pred_df to:", pred_fp)
    print("saved daily models to:", model_dir)


if __name__ == "__main__":
    main()