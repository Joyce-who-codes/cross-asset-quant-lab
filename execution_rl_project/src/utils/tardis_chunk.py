from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.project_paths import resolve_data_root


CHUNK_ROOT = resolve_data_root("TARDIS_CHUNK_ROOT", "tardis_chunks")


def make_day_list(start_day: str, end_day: str) -> list[str]:
    dates = pd.date_range(start=start_day, end=end_day, freq="D")
    return [d.strftime("%Y-%m-%d") for d in dates]


def build_chunk_paths(
    symbol: str,
    start_day: str,
    end_day: str,
    chunk_hours: int = 6,
    chunk_root: str | Path | None = None,
) -> list[dict[str, str]]:
    if chunk_hours <= 0 or 24 % chunk_hours != 0:
        raise ValueError("chunk_hours must be a positive divisor of 24")

    symbol = symbol.upper()
    root = Path(chunk_root) if chunk_root is not None else CHUNK_ROOT
    root = root / symbol

    out: list[dict[str, str]] = []
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
                    {
                        "chunk": chunk,
                        "book_path": str(book_path),
                        "trade_path": str(trade_path),
                        "snapshot_path": str(snapshot_path),
                        "meta_path": str(meta_path),
                    }
                )

    if not out:
        raise FileNotFoundError(
            f"No chunk parquet files found for {symbol} in [{start_day}, {end_day}] under {root}"
        )

    return out
