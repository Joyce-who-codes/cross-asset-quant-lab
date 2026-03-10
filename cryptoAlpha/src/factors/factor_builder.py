from __future__ import annotations

from typing import Callable
import numpy as np
import pandas as pd

from src.factors.factor_utils import (
    ensure_panel_sorted,
    require_columns,
    pct_change_by_symbol,
    rolling_mean_by_symbol,
    rolling_zscore_by_symbol,
)


def factor_mom_24h(panel_df: pd.DataFrame) -> pd.DataFrame:
    df = ensure_panel_sorted(panel_df)
    require_columns(df, ["close"])
    out = df[["datetime", "symbol"]].copy()
    out["mom_24h"] = pct_change_by_symbol(df, "close", 24)
    return out


def factor_rev_6h(panel_df: pd.DataFrame) -> pd.DataFrame:
    df = ensure_panel_sorted(panel_df)
    require_columns(df, ["close"])
    out = df[["datetime", "symbol"]].copy()
    out["rev_6h"] = -pct_change_by_symbol(df, "close", 6)
    return out


def factor_funding_z_24(panel_df: pd.DataFrame) -> pd.DataFrame:
    df = ensure_panel_sorted(panel_df)
    require_columns(df, ["funding_close"])
    out = df[["datetime", "symbol"]].copy()
    out["funding_z_24"] = rolling_zscore_by_symbol(df, "funding_close", 24, min_periods=12)
    return out


def factor_oi_chg_24h(panel_df: pd.DataFrame) -> pd.DataFrame:
    df = ensure_panel_sorted(panel_df)
    require_columns(df, ["oi_close"])
    out = df[["datetime", "symbol"]].copy()
    out["oi_chg_24h"] = pct_change_by_symbol(df, "oi_close", 24)
    return out


def factor_taker_imbalance(panel_df: pd.DataFrame) -> pd.DataFrame:
    df = ensure_panel_sorted(panel_df)
    require_columns(df, ["taker_buy_volume_usd", "taker_sell_volume_usd"])
    out = df[["datetime", "symbol"]].copy()

    buy = df["taker_buy_volume_usd"].astype(float)
    sell = df["taker_sell_volume_usd"].astype(float)
    out["taker_imbalance"] = (buy - sell) / (buy + sell + 1e-12)
    return out


def factor_ls_ratio_z_24(panel_df: pd.DataFrame) -> pd.DataFrame:
    df = ensure_panel_sorted(panel_df)
    require_columns(df, ["global_account_long_short_ratio"])
    out = df[["datetime", "symbol"]].copy()
    out["ls_ratio_z_24"] = rolling_zscore_by_symbol(
        df, "global_account_long_short_ratio", 24, min_periods=12
    )
    return out


def factor_active_community_count_z_24(panel_df: pd.DataFrame) -> pd.DataFrame:
    df = ensure_panel_sorted(panel_df)
    require_columns(df, ["active_community_count"])
    out = df[["datetime", "symbol"]].copy()
    out["active_community_count_z_24"] = rolling_zscore_by_symbol(
        df, "active_community_count", 24, min_periods=12
    )
    return out


def factor_volume_ratio_24(panel_df: pd.DataFrame) -> pd.DataFrame:
    df = ensure_panel_sorted(panel_df)
    require_columns(df, ["volume_usd"])
    out = df[["datetime", "symbol"]].copy()

    vol_ma_24 = rolling_mean_by_symbol(df, "volume_usd", 24, min_periods=12)
    out["volume_ratio_24"] = df["volume_usd"] / (vol_ma_24 + 1e-12)
    return out


def factor_close_to_ma24(panel_df: pd.DataFrame) -> pd.DataFrame:
    df = ensure_panel_sorted(panel_df)
    require_columns(df, ["close"])
    out = df[["datetime", "symbol"]].copy()

    ma_24 = rolling_mean_by_symbol(df, "close", 24, min_periods=12)
    out["close_to_ma24"] = df["close"] / (ma_24 + 1e-12) - 1.0
    return out


FACTOR_REGISTRY: dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    "mom_24h": factor_mom_24h,
    "rev_6h": factor_rev_6h,
    "funding_z_24": factor_funding_z_24,
    "oi_chg_24h": factor_oi_chg_24h,
    "taker_imbalance": factor_taker_imbalance,
    "ls_ratio_z_24": factor_ls_ratio_z_24,
    "active_community_count_z_24": factor_active_community_count_z_24,
    "volume_ratio_24": factor_volume_ratio_24,
    "close_to_ma24": factor_close_to_ma24,
}


class FactorBuilder:
    def __init__(self, factor_registry: dict[str, Callable[[pd.DataFrame], pd.DataFrame]] | None = None) -> None:
        self.factor_registry = factor_registry or FACTOR_REGISTRY

    def list_factors(self) -> list[str]:
        return sorted(self.factor_registry.keys())

    def compute_one(self, panel_df: pd.DataFrame, factor_name: str) -> pd.DataFrame:
        if factor_name not in self.factor_registry:
            raise ValueError(f"Unknown factor: {factor_name}")
        return self.factor_registry[factor_name](panel_df)

    def compute_many(self, panel_df: pd.DataFrame, factor_names: list[str]) -> pd.DataFrame:
        factor_names = list(dict.fromkeys(factor_names))
        if not factor_names:
            return ensure_panel_sorted(panel_df)[["datetime", "symbol"]].copy()

        merged: pd.DataFrame | None = None
        for factor_name in factor_names:
            df_factor = self.compute_one(panel_df, factor_name)
            if merged is None:
                merged = df_factor
            else:
                merged = merged.merge(df_factor, on=["datetime", "symbol"], how="outer")

        assert merged is not None
        merged = merged.sort_values(["datetime", "symbol"]).reset_index(drop=True)
        return merged
