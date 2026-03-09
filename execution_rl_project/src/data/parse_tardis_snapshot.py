from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def _snapshot_columns() -> list[str]:
    cols = ["exchange", "symbol", "timestamp", "local_timestamp"]
    for i in range(25):
        cols.extend(
            [
                f"asks[{i}].price",
                f"asks[{i}].amount",
                f"bids[{i}].price",
                f"bids[{i}].amount",
            ]
        )
    return cols


SNAPSHOT_COLUMNS = _snapshot_columns()


def load_snapshot25_csv(
    path: str | Path,
    symbol: Optional[str] = None,
) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path)

    missing = [c for c in SNAPSHOT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"snapshot_25 csv missing columns: {missing[:10]} ...")

    if symbol is not None:
        df = df[df["symbol"] == symbol].copy()

    df["timestamp"] = df["timestamp"].astype("int64")
    df["local_timestamp"] = df["local_timestamp"].astype("int64")

    price_cols = [c for c in df.columns if c.endswith(".price")]
    amount_cols = [c for c in df.columns if c.endswith(".amount")]

    for c in price_cols + amount_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.reset_index(drop=True)


def snapshot_row_to_books(row: pd.Series) -> tuple[dict[float, float], dict[float, float]]:
    bids: dict[float, float] = {}
    asks: dict[float, float] = {}

    for i in range(25):
        ask_p = row.get(f"asks[{i}].price")
        ask_q = row.get(f"asks[{i}].amount")
        bid_p = row.get(f"bids[{i}].price")
        bid_q = row.get(f"bids[{i}].amount")

        if pd.notna(ask_p) and pd.notna(ask_q) and float(ask_q) > 0:
            asks[float(ask_p)] = float(ask_q)
        if pd.notna(bid_p) and pd.notna(bid_q) and float(bid_q) > 0:
            bids[float(bid_p)] = float(bid_q)

    return bids, asks