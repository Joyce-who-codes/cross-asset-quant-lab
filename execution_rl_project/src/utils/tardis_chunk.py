from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


CHUNK_ROOT = Path("/home/joyce/projects/data/raw/tardis_chunks")


@dataclass
class ChunkPaths:
    chunk: str
    book_path: str
    trade_path: str
    snapshot_path: str
    meta_path: str


def make_day_list(start_day: str, end_day: str) -> list[str]:
    dates = pd.date_range(start=start_day, end=end_day, freq="D")
    return [d.strftime("%Y-%m-%d") for d in dates]


def build_chunk_paths(
    symbol: str,
    start_day: str,
    end_day: str,
    chunk_hours: int = 6,
) -> list[ChunkPaths]:
    if chunk_hours <= 0 or 24 % chunk_hours != 0:
        raise ValueError("chunk_hours must be a positive divisor of 24")

    symbol = symbol.upper()
    root = CHUNK_ROOT / symbol

    out: list[ChunkPaths] = []
    for day in make_day_list(start_day, end_day):
        for hour in range(0, 24, chunk_hours):
            chunk = f"{day}_{hour:02d}"
            chunk_dir = root / chunk
            book_path = chunk_dir / "book.parquet"
            trade_path = chunk_dir / "trades.parquet"
            snapshot_path = chunk_dir / "snapshot.parquet"
            meta_path = chunk_dir / "meta.json"

            if book_path.exists() and trade_path.exists() and snapshot_path.exists() and meta_path.exists():
                out.append(
                    ChunkPaths(
                        chunk=chunk,
                        book_path=str(book_path),
                        trade_path=str(trade_path),
                        snapshot_path=str(snapshot_path),
                        meta_path=str(meta_path),
                    )
                )

    if not out:
        raise FileNotFoundError(
            f"No chunk parquet files found for {symbol} in [{start_day}, {end_day}]"
        )

    return out