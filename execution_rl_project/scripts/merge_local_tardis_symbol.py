from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable
import gzip

import pandas as pd

from src.utils.io import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge local Tardis csv files by symbol")
    parser.add_argument("--symbol", required=True, help="Trading symbol, e.g. APTUSDT")
    parser.add_argument(
        "--config",
        default="configs/batch_data.yaml",
        help="Path to batch data config yaml",
    )
    return parser.parse_args()


def ensure_dirs(dst_book: Path, dst_trades: Path, dst_snapshot: Path) -> None:
    dst_book.mkdir(parents=True, exist_ok=True)
    dst_trades.mkdir(parents=True, exist_ok=True)
    dst_snapshot.mkdir(parents=True, exist_ok=True)


def make_day_list(start_day: str, end_day: str) -> list[str]:
    dates = pd.date_range(start=start_day, end=end_day, freq="D")
    return [d.strftime("%Y-%m-%d") for d in dates]


def _is_gzip_file(path: Path) -> bool:
    with open(path, "rb") as f:
        magic = f.read(2)
    return magic == b"\x1f\x8b"


def _find_existing_file(root: Path, day: str, symbol: str) -> Path:
    candidates = [
        root / day / f"{symbol}.csv",
        root / day / f"{symbol}.csv.gz",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"missing file for {symbol} on {day}, checked: {candidates}")


def _open_text_auto(path: Path):
    if _is_gzip_file(path):
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return open(path, "rt", encoding="utf-8", newline="")


def _open_write_auto(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "wt", encoding="utf-8", newline="")
    return open(path, "wt", encoding="utf-8", newline="")


def stream_merge_csvs(
    input_paths: Iterable[Path],
    output_path: Path,
) -> None:
    input_paths = list(input_paths)
    if not input_paths:
        raise ValueError("input_paths is empty")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_rows_written = 0
    header_written = False

    with _open_write_auto(output_path) as fout:
        for i, path in enumerate(input_paths):
            with _open_text_auto(path) as fin:
                header = fin.readline()
                if not header:
                    print(f"[warn] empty file skipped: {path}")
                    continue

                if not header_written:
                    fout.write(header)
                    header_written = True

                row_count = 0
                for line in fin:
                    fout.write(line)
                    row_count += 1

                total_rows_written += row_count
                print(f"[merge] {i + 1}/{len(input_paths)} {path} rows={row_count}")

    print(f"[done] wrote {total_rows_written} data rows to {output_path}")


def save_split(
    split_name: str,
    symbol: str,
    start_day: str,
    end_day: str,
    src_book: Path,
    src_trades: Path,
    src_snapshot: Path,
    dst_book: Path,
    dst_trades: Path,
    dst_snapshot: Path,
) -> None:
    days = make_day_list(start_day, end_day)

    book_paths = [_find_existing_file(src_book, day, symbol) for day in days]
    trade_paths = [_find_existing_file(src_trades, day, symbol) for day in days]
    snapshot_paths = [_find_existing_file(src_snapshot, day, symbol) for day in days]

    suffix = f"{symbol}_{split_name}_{start_day}_{end_day}"

    book_out = dst_book / f"{suffix}.csv"
    trade_out = dst_trades / f"{suffix}.csv"
    snapshot_out = dst_snapshot / f"{suffix}.csv.gz"

    print(f"\n=== merging {split_name} / book ===")
    stream_merge_csvs(book_paths, book_out)

    print(f"\n=== merging {split_name} / trades ===")
    stream_merge_csvs(trade_paths, trade_out)

    print(f"\n=== merging {split_name} / snapshot ===")
    stream_merge_csvs(snapshot_paths, snapshot_out)

    print(f"\n=== saved {split_name} ===")
    print(book_out)
    print(trade_out)
    print(snapshot_out)


def main() -> None:
    args = parse_args()

    symbol = args.symbol.upper()
    data_cfg = load_yaml(args.config)
    local_root = Path(data_cfg["local_root"])

    src_book = local_root / symbol / "incremental_book_L2"
    src_trades = local_root / symbol / "trades"
    src_snapshot = local_root / symbol / "snapshot_25"

    dst_book = local_root / symbol / "merged" / "incremental_book_L2"
    dst_trades = local_root / symbol / "merged" / "trades"
    dst_snapshot = local_root / symbol / "merged" / "snapshot_25"

    ensure_dirs(dst_book, dst_trades, dst_snapshot)

    for split_name, split_cfg in data_cfg["splits"].items():
        save_split(
            split_name=split_name,
            symbol=symbol,
            start_day=split_cfg["start_day"],
            end_day=split_cfg["end_day"],
            src_book=src_book,
            src_trades=src_trades,
            src_snapshot=src_snapshot,
            dst_book=dst_book,
            dst_trades=dst_trades,
            dst_snapshot=dst_snapshot,
        )


if __name__ == "__main__":
    main()