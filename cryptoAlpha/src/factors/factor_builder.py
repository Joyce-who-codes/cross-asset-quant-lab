from __future__ import annotations

from typing import Callable
import numpy as np
import pandas as pd

from src.factors.factor_utils import (
    ensure_panel_sorted,
    require_columns,
    pct_change_by_symbol,
    rolling_mean_by_symbol,
    rolling_std_by_symbol,
    rolling_zscore_by_symbol,
)

EPS = 1e-12


# =========================
# Momentum / Trend factors
# =========================

def factor_mom_6h(panel_df: pd.DataFrame) -> pd.DataFrame:
    df = ensure_panel_sorted(panel_df)
    require_columns(df, ["close"])
    out = df[["datetime", "symbol"]].copy()
    out["mom_6h"] = pct_change_by_symbol(df, "close", 6)
    return out


def factor_mom_24h(panel_df: pd.DataFrame) -> pd.DataFrame:
    df = ensure_panel_sorted(panel_df)
    require_columns(df, ["close"])
    out = df[["datetime", "symbol"]].copy()
    out["mom_24h"] = pct_change_by_symbol(df, "close", 24)
    return out


def factor_ema_ratio_24(panel_df: pd.DataFrame) -> pd.DataFrame:
    df = ensure_panel_sorted(panel_df)
    require_columns(df, ["close"])
    out = df[["datetime", "symbol"]].copy()

    ema_24 = (
        df.groupby("symbol")["close"]
        .transform(lambda s: s.ewm(span=24, adjust=False, min_periods=12).mean())
    )
    out["ema_ratio_24"] = df["close"] / (ema_24 + EPS) - 1.0
    return out


def factor_ma_gap_24_72(panel_df: pd.DataFrame) -> pd.DataFrame:
    df = ensure_panel_sorted(panel_df)
    require_columns(df, ["close"])
    out = df[["datetime", "symbol"]].copy()

    ma_24 = rolling_mean_by_symbol(df, "close", 24, min_periods=12)
    ma_72 = rolling_mean_by_symbol(df, "close", 72, min_periods=36)
    out["ma_gap_24_72"] = ma_24 / (ma_72 + EPS) - 1.0
    return out


def factor_close_to_high_24(panel_df: pd.DataFrame) -> pd.DataFrame:
    df = ensure_panel_sorted(panel_df)
    require_columns(df, ["close", "high"])
    out = df[["datetime", "symbol"]].copy()

    rolling_high_24 = (
        df.groupby("symbol")["high"]
        .rolling(24, min_periods=12)
        .max()
        .reset_index(level=0, drop=True)
    )
    out["close_to_high_24"] = df["close"] / (rolling_high_24 + EPS) - 1.0
    return out


def factor_trend_strength_12_48(panel_df: pd.DataFrame) -> pd.DataFrame:
    df = ensure_panel_sorted(panel_df)
    require_columns(df, ["close"])
    out = df[["datetime", "symbol"]].copy()

    ema_12 = (
        df.groupby("symbol")["close"]
        .transform(lambda s: s.ewm(span=12, adjust=False, min_periods=6).mean())
    )
    ema_48 = (
        df.groupby("symbol")["close"]
        .transform(lambda s: s.ewm(span=48, adjust=False, min_periods=24).mean())
    )
    out["trend_strength_12_48"] = (ema_12 - ema_48) / (ema_48 + EPS)
    return out


# =========================
# Volatility factors
# =========================

def factor_realized_vol_24(panel_df: pd.DataFrame) -> pd.DataFrame:
    df = ensure_panel_sorted(panel_df)
    require_columns(df, ["close"])
    out = df[["datetime", "symbol"]].copy()

    ret_1h = pct_change_by_symbol(df, "close", 1)
    rv_24 = (
        ret_1h.groupby(df["symbol"])
        .rolling(24, min_periods=12)
        .apply(lambda x: np.sqrt(np.sum(np.square(x))), raw=True)
        .reset_index(level=0, drop=True)
    )
    out["realized_vol_24"] = rv_24
    return out


def factor_volatility_ratio_6_24(panel_df: pd.DataFrame) -> pd.DataFrame:
    df = ensure_panel_sorted(panel_df)
    require_columns(df, ["close"])
    out = df[["datetime", "symbol"]].copy()

    ret_1h = pct_change_by_symbol(df, "close", 1)
    vol_6 = (
        ret_1h.groupby(df["symbol"])
        .rolling(6, min_periods=3)
        .std()
        .reset_index(level=0, drop=True)
    )
    vol_24 = (
        ret_1h.groupby(df["symbol"])
        .rolling(24, min_periods=12)
        .std()
        .reset_index(level=0, drop=True)
    )
    out["volatility_ratio_6_24"] = vol_6 / (vol_24 + EPS)
    return out


def factor_price_range(panel_df: pd.DataFrame) -> pd.DataFrame:
    df = ensure_panel_sorted(panel_df)
    require_columns(df, ["high", "low", "close"])
    out = df[["datetime", "symbol"]].copy()
    out["price_range"] = (df["high"] - df["low"]) / (df["close"] + EPS)
    return out


def factor_atr_like_14(panel_df: pd.DataFrame) -> pd.DataFrame:
    df = ensure_panel_sorted(panel_df)
    require_columns(df, ["high", "low", "close"])
    out = df[["datetime", "symbol"]].copy()

    prev_close = df.groupby("symbol")["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr_14 = (
        tr.groupby(df["symbol"])
        .transform(lambda s: s.ewm(span=14, adjust=False, min_periods=7).mean())
    )
    out["atr_like_14"] = atr_14 / (df["close"] + EPS)
    return out


# =========================
# Volume / Liquidity factors
# =========================

def factor_volume_ratio_24(panel_df: pd.DataFrame) -> pd.DataFrame:
    df = ensure_panel_sorted(panel_df)
    require_columns(df, ["volume_usd"])
    out = df[["datetime", "symbol"]].copy()

    vol_ma_24 = rolling_mean_by_symbol(df, "volume_usd", 24, min_periods=12)
    out["volume_ratio_24"] = df["volume_usd"] / (vol_ma_24 + EPS)
    return out


def factor_volume_momentum_6_24(panel_df: pd.DataFrame) -> pd.DataFrame:
    df = ensure_panel_sorted(panel_df)
    require_columns(df, ["volume_usd"])
    out = df[["datetime", "symbol"]].copy()

    vol_ma_6 = rolling_mean_by_symbol(df, "volume_usd", 6, min_periods=3)
    vol_ma_24 = rolling_mean_by_symbol(df, "volume_usd", 24, min_periods=12)
    out["volume_momentum_6_24"] = vol_ma_6 / (vol_ma_24 + EPS)
    return out


def factor_amihud_illiquidity_24(panel_df: pd.DataFrame) -> pd.DataFrame:
    df = ensure_panel_sorted(panel_df)
    require_columns(df, ["close", "volume_usd"])
    out = df[["datetime", "symbol"]].copy()

    ret_1h_abs = pct_change_by_symbol(df, "close", 1).abs()
    illiq_raw = ret_1h_abs / (df["volume_usd"] + EPS)
    illiq_24 = (
        illiq_raw.groupby(df["symbol"])
        .rolling(24, min_periods=12)
        .mean()
        .reset_index(level=0, drop=True)
    )
    out["amihud_illiquidity_24"] = illiq_24
    return out


def factor_turnover_volatility_24(panel_df: pd.DataFrame) -> pd.DataFrame:
    df = ensure_panel_sorted(panel_df)
    require_columns(df, ["volume_usd"])
    out = df[["datetime", "symbol"]].copy()

    out["turnover_volatility_24"] = rolling_std_by_symbol(
        df, "volume_usd", 24, min_periods=12
    )
    return out


# =========================
# Derivatives factors
# =========================

def factor_funding_z_24(panel_df: pd.DataFrame) -> pd.DataFrame:
    df = ensure_panel_sorted(panel_df)
    require_columns(df, ["funding_close"])
    out = df[["datetime", "symbol"]].copy()

    out["funding_z_24"] = rolling_zscore_by_symbol(
        df, "funding_close", 24, min_periods=12
    )
    return out


def factor_oi_change_24h(panel_df: pd.DataFrame) -> pd.DataFrame:
    df = ensure_panel_sorted(panel_df)
    require_columns(df, ["oi_close"])
    out = df[["datetime", "symbol"]].copy()

    out["oi_change_24h"] = pct_change_by_symbol(df, "oi_close", 24)
    return out


def factor_oi_volume_ratio(panel_df: pd.DataFrame) -> pd.DataFrame:
    df = ensure_panel_sorted(panel_df)
    require_columns(df, ["oi_close", "volume_usd"])
    out = df[["datetime", "symbol"]].copy()

    out["oi_volume_ratio"] = df["oi_close"] / (df["volume_usd"] + EPS)
    return out


def factor_long_short_ratio_z_24(panel_df: pd.DataFrame) -> pd.DataFrame:
    df = ensure_panel_sorted(panel_df)
    require_columns(df, ["global_account_long_short_ratio"])
    out = df[["datetime", "symbol"]].copy()

    out["long_short_ratio_z_24"] = rolling_zscore_by_symbol(
        df, "global_account_long_short_ratio", 24, min_periods=12
    )
    return out


def factor_taker_imbalance(panel_df: pd.DataFrame) -> pd.DataFrame:
    df = ensure_panel_sorted(panel_df)
    require_columns(df, ["taker_buy_volume_usd", "taker_sell_volume_usd"])
    out = df[["datetime", "symbol"]].copy()

    buy = df["taker_buy_volume_usd"].astype(float)
    sell = df["taker_sell_volume_usd"].astype(float)
    out["taker_imbalance"] = (buy - sell) / (buy + sell + EPS)
    return out


# =========================
# Sentiment factors
# =========================

def factor_active_community_count_z_24(panel_df: pd.DataFrame) -> pd.DataFrame:
    df = ensure_panel_sorted(panel_df)
    require_columns(df, ["active_community_count"])
    out = df[["datetime", "symbol"]].copy()

    out["active_community_count_z_24"] = rolling_zscore_by_symbol(
        df, "active_community_count", 24, min_periods=12
    )
    return out


# =========================
# Registry
# =========================

FACTOR_REGISTRY: dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    # Momentum / Trend
    "mom_6h": factor_mom_6h,
    "mom_24h": factor_mom_24h,
    "ema_ratio_24": factor_ema_ratio_24,
    "ma_gap_24_72": factor_ma_gap_24_72,
    "close_to_high_24": factor_close_to_high_24,
    "trend_strength_12_48": factor_trend_strength_12_48,

    # Volatility
    "realized_vol_24": factor_realized_vol_24,
    "volatility_ratio_6_24": factor_volatility_ratio_6_24,
    "price_range": factor_price_range,
    "atr_like_14": factor_atr_like_14,

    # Volume / Liquidity
    "volume_ratio_24": factor_volume_ratio_24,
    "volume_momentum_6_24": factor_volume_momentum_6_24,
    "amihud_illiquidity_24": factor_amihud_illiquidity_24,
    "turnover_volatility_24": factor_turnover_volatility_24,

    # Derivatives
    "funding_z_24": factor_funding_z_24,
    "oi_change_24h": factor_oi_change_24h,
    "oi_volume_ratio": factor_oi_volume_ratio,
    "long_short_ratio_z_24": factor_long_short_ratio_z_24,
    "taker_imbalance": factor_taker_imbalance,

    # Sentiment
    "active_community_count_z_24": factor_active_community_count_z_24,
}


class FactorBuilder:
    def __init__(
        self,
        factor_registry: dict[str, Callable[[pd.DataFrame], pd.DataFrame]] | None = None,
    ) -> None:
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