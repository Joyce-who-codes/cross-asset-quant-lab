"""Coinglass data loader."""
from __future__ import annotations

from pathlib import Path
from typing import Optional
import pandas as pd

try:
    from src.data.registry import DataRegistry
    from src.data.loaders.base import read_symbol_csv_dir
    from src.data.schema import standardize_dataset_df
except ImportError:
    from registry import DataRegistry
    from loaders.base import read_symbol_csv_dir
    from schema import standardize_dataset_df


class CoinglassLoader:
    """
    Coinglass NORMAL loader

    Supported path pattern:
    /COINGLASS_v6/NORMAL/{dataset}/{freq}/{exchange}/{symbol}/*.csv
    """

    def __init__(self, registry: Optional[DataRegistry] = None) -> None:
        self.registry = registry or DataRegistry()

    @property
    def root(self) -> Path:
        return self.registry.coinglass_root

    def list_datasets(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted([p.name for p in self.root.iterdir() if p.is_dir()])

    def list_symbols(
        self,
        dataset: str,
        freq: str = "1h",
        exchange: str = "Binance",
    ) -> list[str]:
        ds_root = self.root / dataset / freq / exchange
        if not ds_root.exists():
            return []
        return sorted([p.name for p in ds_root.iterdir() if p.is_dir()])

    def get_symbol_dir(
        self,
        dataset: str,
        symbol: str,
        freq: str = "1h",
        exchange: str = "Binance",
    ) -> Path:
        return self.root / dataset / freq / exchange / symbol

    def exists(
        self,
        dataset: str,
        symbol: str,
        freq: str = "1h",
        exchange: str = "Binance",
    ) -> bool:
        return self.get_symbol_dir(dataset, symbol, freq, exchange).exists()

    def load_symbol(
        self,
        dataset: str,
        symbol: str,
        freq: str = "1h",
        exchange: str = "Binance",
        start_time: str | None = None,
        end_time: str | None = None,
    ) -> pd.DataFrame:
        symbol_dir = self.get_symbol_dir(
            dataset=dataset,
            symbol=symbol,
            freq=freq,
            exchange=exchange,
        )
        raw = read_symbol_csv_dir(symbol_dir)
        if raw.empty:
            return pd.DataFrame(columns=["datetime", "symbol"])

        df = standardize_dataset_df(
            raw,
            source="coinglass",
            dataset=dataset,
            symbol=symbol,
        )
        if df.empty:
            return df

        if start_time is not None:
            df = df[df["datetime"] >= pd.Timestamp(start_time)]
        if end_time is not None:
            df = df[df["datetime"] <= pd.Timestamp(end_time)]

        return df.sort_values(["datetime", "symbol"]).reset_index(drop=True)

    def load_many_symbols(
        self,
        dataset: str,
        symbols: list[str],
        freq: str = "1h",
        exchange: str = "Binance",
        start_time: str | None = None,
        end_time: str | None = None,
        skip_missing: bool = True,
    ) -> pd.DataFrame:
        dfs: list[pd.DataFrame] = []

        for symbol in symbols:
            if skip_missing and not self.exists(dataset, symbol, freq, exchange):
                continue

            df = self.load_symbol(
                dataset=dataset,
                symbol=symbol,
                freq=freq,
                exchange=exchange,
                start_time=start_time,
                end_time=end_time,
            )
            if not df.empty:
                dfs.append(df)

        if not dfs:
            return pd.DataFrame(columns=["datetime", "symbol"])

        out = pd.concat(dfs, axis=0, ignore_index=True)
        out = out.sort_values(["datetime", "symbol"]).reset_index(drop=True)
        return out