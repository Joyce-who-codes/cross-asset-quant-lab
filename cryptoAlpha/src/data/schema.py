from __future__ import annotations

from typing import Any
import pandas as pd


SCHEMA_SPECS: dict[str, dict[str, dict[str, Any]]] = {
    "coinglass": {
        "FUTURES_PRICE_HISTORY": {
            "time_candidates": ["timestamp"],
            "column_map": {
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume_usd": "volume_usd",
            },
        },
        "FUTURES_FUNDING_RATE_HISTORY": {
            "time_candidates": ["timestamp"],
            "column_map": {
                "open": "funding_open",
                "high": "funding_high",
                "low": "funding_low",
                "close": "funding_close",
            },
        },
        "FUTURES_OPEN_INTEREST": {
            "time_candidates": ["timestamp", "time"],
            "column_map": {
                "open": "oi_open",
                "high": "oi_high",
                "low": "oi_low",
                "close": "oi_close",
            },
        },
        "FUTURES_TAKER_BUY_SELL_VOLUME": {
            "time_candidates": ["timestamp", "time"],
            "column_map": {
                "taker_buy_volume_usd": "taker_buy_volume_usd",
                "taker_sell_volume_usd": "taker_sell_volume_usd",
            },
        },
        "FUTURES_GLOBAL_LS_ACCOUNT_RATIO": {
            "time_candidates": ["timestamp", "time"],
            "column_map": {
                "global_account_long_percent": "global_account_long_percent",
                "global_account_short_percent": "global_account_short_percent",
                "global_account_long_short_ratio": "global_account_long_short_ratio",
            },
        },
    },
    "coingecko": {
        "COIN_MARKET_CHART/hourly": {
            "time_candidates": ["normal_time", "timestamp", "time"],
            "column_map": {
                "price": "price",
                "market_cap": "market_cap",
                "total_volume": "total_volume",
            },
        },
    },
    "cryptoracle": {
        "active_community_count": {
            "time_candidates": ["end_time", "start_time"],
            "column_map": {
                "value": "value",
            },
        },
    },
}


CRYPTORACLE_GENERIC_SPEC: dict[str, Any] = {
    "time_candidates": ["end_time", "start_time", "timestamp"],
    "column_map": {
        "value": "value",
    },
}


def _pick_existing_column(df: pd.DataFrame, candidates: list[str]) -> str:
    cols_lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in cols_lower:
            return cols_lower[c.lower()]
    raise ValueError(f"Cannot find any of {candidates} in columns={list(df.columns)}")


def standardize_dataset_df(
    df: pd.DataFrame,
    source: str,
    dataset: str,
    symbol: str,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["datetime", "symbol"])

    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    if source not in SCHEMA_SPECS:
        raise ValueError(f"Unsupported schema: source={source}, dataset={dataset}")

    if dataset in SCHEMA_SPECS[source]:
        spec = SCHEMA_SPECS[source][dataset]
    elif source == "cryptoracle":
        spec = CRYPTORACLE_GENERIC_SPEC
    else:
        raise ValueError(f"Unsupported schema: source={source}, dataset={dataset}")

    time_col = _pick_existing_column(df, spec["time_candidates"])
    out = pd.DataFrame()
    out["datetime"] = (
        pd.to_datetime(df[time_col], errors="coerce", utc=True)
        .dt.tz_localize(None)
    )
    out["symbol"] = symbol

    for raw_col, std_col in spec["column_map"].items():
        if raw_col in df.columns:
            out[std_col] = pd.to_numeric(df[raw_col], errors="coerce")
        else:
            out[std_col] = pd.NA

    out = out[out["datetime"].notna()].copy()
    out = out.sort_values(["datetime", "symbol"])
    out = out.drop_duplicates(subset=["datetime", "symbol"], keep="last")
    out = out.reset_index(drop=True)
    return out