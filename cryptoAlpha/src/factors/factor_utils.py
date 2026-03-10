from __future__ import annotations

from typing import Iterable
import numpy as np
import pandas as pd


EPS = 1e-12


def ensure_panel_sorted(df: pd.DataFrame) -> pd.DataFrame:
    required = {"datetime", "symbol"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()
    out["datetime"] = pd.to_datetime(out["datetime"])
    out["symbol"] = out["symbol"].astype(str)
    out = out.sort_values(["symbol", "datetime"]).reset_index(drop=True)
    return out


def require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def pct_change_by_symbol(df: pd.DataFrame, col: str, periods: int) -> pd.Series:
    require_columns(df, [col])
    return df.groupby("symbol")[col].pct_change(periods)


def diff_by_symbol(df: pd.DataFrame, col: str, periods: int = 1) -> pd.Series:
    require_columns(df, [col])
    return df.groupby("symbol")[col].diff(periods)


def rolling_mean_by_symbol(df: pd.DataFrame, col: str, window: int, min_periods: int | None = None) -> pd.Series:
    require_columns(df, [col])
    min_periods = min_periods or max(1, window // 2)
    return (
        df.groupby("symbol")[col]
        .rolling(window, min_periods=min_periods)
        .mean()
        .reset_index(level=0, drop=True)
    )


def rolling_std_by_symbol(df: pd.DataFrame, col: str, window: int, min_periods: int | None = None) -> pd.Series:
    require_columns(df, [col])
    min_periods = min_periods or max(1, window // 2)
    return (
        df.groupby("symbol")[col]
        .rolling(window, min_periods=min_periods)
        .std()
        .reset_index(level=0, drop=True)
    )


def rolling_zscore_by_symbol(df: pd.DataFrame, col: str, window: int, min_periods: int | None = None) -> pd.Series:
    x = df[col]
    mean = rolling_mean_by_symbol(df, col, window, min_periods)
    std = rolling_std_by_symbol(df, col, window, min_periods)
    return (x - mean) / (std + EPS)


def cs_rank(df: pd.DataFrame, col: str, pct: bool = True) -> pd.Series:
    require_columns(df, [col])
    return df.groupby("datetime")[col].rank(pct=pct, method="average")


def cs_zscore(df: pd.DataFrame, col: str) -> pd.Series:
    require_columns(df, [col])
    g = df.groupby("datetime")[col]
    mean = g.transform("mean")
    std = g.transform("std")
    return (df[col] - mean) / (std + EPS)


def winsorize_by_date(
    df: pd.DataFrame,
    col: str,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
) -> pd.Series:
    require_columns(df, [col])

    def _clip(s: pd.Series) -> pd.Series:
        lo = s.quantile(lower_q)
        hi = s.quantile(upper_q)
        return s.clip(lo, hi)

    return df.groupby("datetime")[col].transform(_clip)


def forward_return(df: pd.DataFrame, price_col: str, horizon: int) -> pd.Series:
    require_columns(df, [price_col])
    future_price = df.groupby("symbol")[price_col].shift(-horizon)
    return future_price / df[price_col] - 1.0


def make_forward_returns(
    panel_df: pd.DataFrame,
    price_col: str = "close",
    horizons: list[int] | None = None,
) -> pd.DataFrame:
    horizons = horizons or [1, 3, 6, 12, 24]
    df = ensure_panel_sorted(panel_df)

    out = df[["datetime", "symbol"]].copy()
    for h in horizons:
        out[f"label_ret_{h}h"] = forward_return(df, price_col=price_col, horizon=h)
    return out
