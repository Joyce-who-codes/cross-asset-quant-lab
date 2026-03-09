from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


BOOK_COLUMNS = [
    "exchange",
    "symbol",
    "timestamp",
    "local_timestamp",
    "is_snapshot",
    "side",
    "price",
    "amount",
]

TRADE_COLUMNS = [
    "exchange",
    "symbol",
    "timestamp",
    "local_timestamp",
    "id",
    "side",
    "price",
    "amount",
]


def _normalize_bool_col(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    return (
        df[col]
        .astype(str)
        .str.lower()
        .map({"true": True, "false": False})
        .fillna(False)
        .astype(bool)
    )


def load_incremental_book_csv(
    path: str | Path,
    symbol: Optional[str] = None,
) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path)

    missing = [c for c in BOOK_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"book csv missing columns: {missing}")

    if symbol is not None:
        df = df[df["symbol"] == symbol].copy()

    df = df[BOOK_COLUMNS].copy()
    df["timestamp"] = df["timestamp"].astype("int64")
    df["local_timestamp"] = df["local_timestamp"].astype("int64")
    df["is_snapshot"] = _normalize_bool_col(df, "is_snapshot")
    df["side"] = df["side"].astype(str).str.lower()
    df["price"] = pd.to_numeric(df["price"], errors="coerce").astype(float)
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").astype(float)

    df["event_type"] = "book"
    df["trade_id"] = pd.NA

    df = df.rename(
        columns={
            "timestamp": "exch_ts",
            "local_timestamp": "local_ts",
        }
    )

    return df[
        [
            "event_type",
            "exchange",
            "symbol",
            "exch_ts",
            "local_ts",
            "is_snapshot",
            "side",
            "price",
            "amount",
            "trade_id",
        ]
    ].reset_index(drop=True)


def load_trades_csv(
    path: str | Path,
    symbol: Optional[str] = None,
) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path)

    missing = [c for c in TRADE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"trades csv missing columns: {missing}")

    if symbol is not None:
        df = df[df["symbol"] == symbol].copy()

    df = df[TRADE_COLUMNS].copy()
    df["timestamp"] = df["timestamp"].astype("int64")
    df["local_timestamp"] = df["local_timestamp"].astype("int64")
    df["id"] = df["id"].astype("int64")
    df["side"] = df["side"].astype(str).str.lower()
    df["price"] = pd.to_numeric(df["price"], errors="coerce").astype(float)
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").astype(float)

    df["event_type"] = "trade"
    df["is_snapshot"] = False

    df = df.rename(
        columns={
            "timestamp": "exch_ts",
            "local_timestamp": "local_ts",
            "id": "trade_id",
        }
    )

    return df[
        [
            "event_type",
            "exchange",
            "symbol",
            "exch_ts",
            "local_ts",
            "is_snapshot",
            "side",
            "price",
            "amount",
            "trade_id",
        ]
    ].reset_index(drop=True)


def _add_encoded_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["event_code"] = (
        out["event_type"]
        .map({"book": 0, "trade": 1})
        .astype("int8")
    )

    out["side_code"] = (
        out["side"]
        .map(
            {
                "bid": 0,
                "ask": 1,
                "buy": 2,
                "sell": 3,
            }
        )
        .astype("int8")
    )

    return out


def merge_book_and_trades(
    book_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    sort_by: str = "exch_ts",
) -> pd.DataFrame:
    if sort_by not in {"exch_ts", "local_ts"}:
        raise ValueError("sort_by must be 'exch_ts' or 'local_ts'")

    df = pd.concat([book_df, trades_df], axis=0, ignore_index=True)

    # 同一时刻先处理 book，再处理 trade
    event_priority = {"book": 0, "trade": 1}
    df["_event_priority"] = df["event_type"].map(event_priority).fillna(9)

    df = df.sort_values(
        by=[sort_by, "_event_priority", "local_ts"],
        kind="mergesort",
    ).reset_index(drop=True)

    df = df.drop(columns=["_event_priority"])
    df = _add_encoded_columns(df)

    return df


def load_merged_events(
    book_path: str | Path,
    trade_path: str | Path,
    symbol: str = "BTCUSDT",
    sort_by: str = "exch_ts",
) -> pd.DataFrame:
    book_df = load_incremental_book_csv(book_path, symbol=symbol)
    trades_df = load_trades_csv(trade_path, symbol=symbol)
    return merge_book_and_trades(book_df, trades_df, sort_by=sort_by)