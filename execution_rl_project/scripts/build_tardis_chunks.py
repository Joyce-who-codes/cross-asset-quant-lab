from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.parse_tardis_csv import NORMALIZED_EVENT_COLUMNS, load_incremental_book_csv, load_trades_csv
from src.data.parse_tardis_snapshot import SNAPSHOT_COLUMNS, load_snapshot25_csv
from src.utils.tardis_chunk import CHUNK_ROOT
from src.utils.tardis_daily import build_daily_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build parquet chunks from raw Tardis csv files")
    parser.add_argument("--symbol", required=True, help="Trading symbol, e.g. BTCUSDT")
    parser.add_argument("--start-day", required=True, help="Inclusive start day, YYYY-MM-DD")
    parser.add_argument("--end-day", required=True, help="Inclusive end day, YYYY-MM-DD")
    parser.add_argument("--chunk-hours", type=int, default=6, help="Chunk size in hours")
    parser.add_argument("--force", action="store_true", help="Overwrite existing chunk parquet files")
    return parser.parse_args()


def to_epoch_us(ts: pd.Timestamp) -> int:
    return int(ts.value // 1_000)


def slice_frame(df: pd.DataFrame, ts_col: str, start_us: int, end_us: int) -> pd.DataFrame:
    return df[(df[ts_col] >= start_us) & (df[ts_col] < end_us)].copy()


def write_chunk(
    symbol: str,
    chunk_dir: Path,
    chunk_name: str,
    start_us: int,
    end_us: int,
    book_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    snapshot_df: pd.DataFrame,
    force: bool,
) -> None:
    chunk_dir.mkdir(parents=True, exist_ok=True)

    book_path = chunk_dir / "book.parquet"
    trades_path = chunk_dir / "trades.parquet"
    snapshot_path = chunk_dir / "snapshot.parquet"
    meta_path = chunk_dir / "meta.json"

    if not force and book_path.exists() and trades_path.exists() and snapshot_path.exists() and meta_path.exists():
        print(f"[skip] chunk exists: {chunk_name}")
        return

    book_df[NORMALIZED_EVENT_COLUMNS].to_parquet(book_path, index=False)
    trades_df[NORMALIZED_EVENT_COLUMNS].to_parquet(trades_path, index=False)
    snapshot_df[SNAPSHOT_COLUMNS].to_parquet(snapshot_path, index=False)

    meta = {
        "symbol": symbol,
        "chunk": chunk_name,
        "start_us": start_us,
        "end_us": end_us,
        "book_rows": int(len(book_df)),
        "trade_rows": int(len(trades_df)),
        "snapshot_rows": int(len(snapshot_df)),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(
        f"[write] {chunk_name} | book={len(book_df)} trade={len(trades_df)} snapshot={len(snapshot_df)}"
    )


def main() -> None:
    args = parse_args()

    symbol = args.symbol.upper()
    chunk_hours = int(args.chunk_hours)
    if chunk_hours <= 0 or 24 % chunk_hours != 0:
        raise ValueError("chunk_hours must be a positive divisor of 24")

    daily_paths = build_daily_paths(symbol=symbol, start_day=args.start_day, end_day=args.end_day)
    symbol_root = CHUNK_ROOT / symbol

    for daily in daily_paths:
        print(f"[load] {daily.day}")
        book_df = load_incremental_book_csv(daily.book_path, symbol=symbol)
        trades_df = load_trades_csv(daily.trade_path, symbol=symbol)
        snapshot_df = load_snapshot25_csv(daily.snapshot_path, symbol=symbol)

        day_start = pd.Timestamp(daily.day, tz="UTC")

        for hour in range(0, 24, chunk_hours):
            chunk_start = day_start + pd.Timedelta(hours=hour)
            chunk_end = chunk_start + pd.Timedelta(hours=chunk_hours)
            start_us = to_epoch_us(chunk_start)
            end_us = to_epoch_us(chunk_end)
            chunk_name = f"{daily.day}_{hour:02d}"

            book_chunk = slice_frame(book_df, "exch_ts", start_us, end_us)
            trades_chunk = slice_frame(trades_df, "exch_ts", start_us, end_us)
            snapshot_chunk = slice_frame(snapshot_df, "timestamp", start_us, end_us)

            if book_chunk.empty and trades_chunk.empty and snapshot_chunk.empty:
                continue

            write_chunk(
                symbol=symbol,
                chunk_dir=symbol_root / chunk_name,
                chunk_name=chunk_name,
                start_us=start_us,
                end_us=end_us,
                book_df=book_chunk,
                trades_df=trades_chunk,
                snapshot_df=snapshot_chunk,
                force=args.force,
            )


if __name__ == "__main__":
    main()