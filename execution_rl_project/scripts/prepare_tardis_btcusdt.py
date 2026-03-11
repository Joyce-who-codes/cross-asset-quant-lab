# filepath: /home/joyce/projects/cross-asset-quant-lab/execution_rl_project/scripts/prepare_tardis_btcusdt.py
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


SYMBOL = "BTCUSDT"

SRC_ROOT = Path("/data/qr/2_TDS/binance-futures")
DST_ROOT = Path("/home/joyce/projects/data/raw/tardis") / SYMBOL

SRC_BOOK = SRC_ROOT / "incremental_book_L2" / SYMBOL
SRC_TRADES = SRC_ROOT / "trades" / SYMBOL
SRC_SNAPSHOT = SRC_ROOT / "book_snapshot_25" / SYMBOL

DST_BOOK = DST_ROOT / "incremental_book_L2"
DST_TRADES = DST_ROOT / "trades"
DST_SNAPSHOT = DST_ROOT / "snapshot_25"


def ensure_dirs() -> None:
    DST_BOOK.mkdir(parents=True, exist_ok=True)
    DST_TRADES.mkdir(parents=True, exist_ok=True)
    DST_SNAPSHOT.mkdir(parents=True, exist_ok=True)


def _filter_time_range(
    df: pd.DataFrame,
    ts_col: str,
    start_ts: int | None = None,
    end_ts: int | None = None,
) -> pd.DataFrame:
    out = df
    if start_ts is not None:
        out = out[out[ts_col] >= start_ts]
    if end_ts is not None:
        out = out[out[ts_col] <= end_ts]
    return out


def _read_csv_auto(path: Path) -> pd.DataFrame:
    if path.suffix == ".gz":
        return pd.read_csv(path, compression="gzip")
    return pd.read_csv(path)


def _resolve_book_path(day: str) -> Path:
    path = SRC_BOOK / day / f"{SYMBOL}.csv"
    if not path.exists():
        raise FileNotFoundError(f"book file not found: {path}")
    return path


def _resolve_trades_path(day: str) -> Path:
    path = SRC_TRADES / day / f"{SYMBOL}.csv"
    if not path.exists():
        raise FileNotFoundError(f"trades file not found: {path}")
    return path


def _resolve_snapshot_path(day: str) -> Path:
    path = SRC_SNAPSHOT / day / f"{SYMBOL}.csv.gz"
    if not path.exists():
        raise FileNotFoundError(f"snapshot file not found: {path}")
    return path


def load_incremental_book(
    days: Iterable[str],
    start_ts: int | None = None,
    end_ts: int | None = None,
) -> pd.DataFrame:
    dfs: list[pd.DataFrame] = []
    for day in days:
        path = _resolve_book_path(day)
        df = _read_csv_auto(path)

        if "timestamp" not in df.columns:
            raise ValueError(f"'timestamp' not found in incremental_book_L2 columns: {path}")

        df = _filter_time_range(df, "timestamp", start_ts=start_ts, end_ts=end_ts)
        dfs.append(df)

        print(f"[book] loaded {day}: rows={len(df)} from {path}")

    if not dfs:
        raise ValueError("no book data loaded")

    out = pd.concat(dfs, ignore_index=True)
    out = out.sort_values("timestamp").reset_index(drop=True)
    print(f"[book] merged rows={len(out)}")
    return out


def load_trades(
    days: Iterable[str],
    start_ts: int | None = None,
    end_ts: int | None = None,
) -> pd.DataFrame:
    dfs: list[pd.DataFrame] = []
    for day in days:
        path = _resolve_trades_path(day)
        df = _read_csv_auto(path)

        if "timestamp" not in df.columns:
            raise ValueError(f"'timestamp' not found in trades columns: {path}")

        df = _filter_time_range(df, "timestamp", start_ts=start_ts, end_ts=end_ts)
        dfs.append(df)

        print(f"[trades] loaded {day}: rows={len(df)} from {path}")

    if not dfs:
        raise ValueError("no trades data loaded")

    out = pd.concat(dfs, ignore_index=True)
    out = out.sort_values("timestamp").reset_index(drop=True)
    print(f"[trades] merged rows={len(out)}")
    return out


def load_snapshot_25(
    days: Iterable[str],
    start_ts: int | None = None,
    end_ts: int | None = None,
) -> pd.DataFrame:
    dfs: list[pd.DataFrame] = []
    for day in days:
        path = _resolve_snapshot_path(day)
        df = _read_csv_auto(path)

        if "timestamp" not in df.columns:
            raise ValueError(f"'timestamp' not found in snapshot_25 columns: {path}")

        df = _filter_time_range(df, "timestamp", start_ts=start_ts, end_ts=end_ts)
        dfs.append(df)

        print(f"[snapshot] loaded {day}: rows={len(df)} from {path}")

    if not dfs:
        raise ValueError("no snapshot data loaded")

    out = pd.concat(dfs, ignore_index=True)
    out = out.sort_values("timestamp").reset_index(drop=True)
    print(f"[snapshot] merged rows={len(out)}")
    return out


def save_outputs(
    book_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    snapshot_df: pd.DataFrame,
    suffix: str = "merged",
) -> None:
    book_path = DST_BOOK / f"{SYMBOL}_{suffix}.csv"
    trades_path = DST_TRADES / f"{SYMBOL}_{suffix}.csv"
    snapshot_path = DST_SNAPSHOT / f"{SYMBOL}_{suffix}.csv.gz"

    book_df.to_csv(book_path, index=False)
    trades_df.to_csv(trades_path, index=False)
    snapshot_df.to_csv(snapshot_path, index=False, compression="gzip")

    print("\n=== Saved Files ===")
    print(f"book:     {book_path}")
    print(f"trades:   {trades_path}")
    print(f"snapshot: {snapshot_path}")


def main() -> None:
    ensure_dirs()

    # ===== user config =====
    days = [
    "2025-12-05",
    "2025-12-06",
    "2025-12-07",]   

    # microsecond timestamps; set to None to keep full-day data
    start_ts: int | None = None
    end_ts: int | None = None

    # output file suffix
    suffix = "2025-12-05_2025-12-07"

    # ===== run =====
    print("=== Prepare Tardis BTCUSDT ===")
    print(f"days={days}")
    print(f"start_ts={start_ts}")
    print(f"end_ts={end_ts}")
    print(f"dst_root={DST_ROOT}")

    book_df = load_incremental_book(days, start_ts=start_ts, end_ts=end_ts)
    trades_df = load_trades(days, start_ts=start_ts, end_ts=end_ts)
    snapshot_df = load_snapshot_25(days, start_ts=start_ts, end_ts=end_ts)

    save_outputs(
        book_df=book_df,
        trades_df=trades_df,
        snapshot_df=snapshot_df,
        suffix=suffix,
    )


if __name__ == "__main__":
    main()