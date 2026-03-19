from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.utils.project_paths import resolve_data_root


DATA_ROOT = resolve_data_root("TARDIS_RAW_ROOT", "tardis")


@dataclass
class DailyPaths:
    day: str
    book_path: str
    trade_path: str
    snapshot_path: str


def make_day_list(start_day: str, end_day: str) -> list[str]:
    dates = pd.date_range(start=start_day, end=end_day, freq="D")
    return [d.strftime("%Y-%m-%d") for d in dates]


def build_daily_paths(
    symbol: str,
    start_day: str,
    end_day: str,
    raw_root: str | Path | None = None,
) -> list[DailyPaths]:
    symbol = symbol.upper()
    root = Path(raw_root) if raw_root is not None else DATA_ROOT
    root = root / symbol

    out: list[DailyPaths] = []
    for day in make_day_list(start_day, end_day):
        book_candidates = [
            root / "incremental_book_L2" / day / f"{symbol}.csv",
            root / "incremental_book_L2" / day / f"{symbol}.csv.gz",
        ]
        trade_candidates = [
            root / "trades" / day / f"{symbol}.csv",
            root / "trades" / day / f"{symbol}.csv.gz",
        ]
        snapshot_candidates = [
            root / "snapshot_25" / day / f"{symbol}.csv.gz",
            root / "snapshot_25" / day / f"{symbol}.csv",
        ]

        book_path = next((str(p) for p in book_candidates if p.exists()), None)
        trade_path = next((str(p) for p in trade_candidates if p.exists()), None)
        snapshot_path = next((str(p) for p in snapshot_candidates if p.exists()), None)

        if book_path and trade_path and snapshot_path:
            out.append(
                DailyPaths(
                    day=day,
                    book_path=book_path,
                    trade_path=trade_path,
                    snapshot_path=snapshot_path,
                )
            )

    if not out:
        raise FileNotFoundError(
            f"No daily Tardis files found for {symbol} in [{start_day}, {end_day}] under {root}"
        )

    return out
