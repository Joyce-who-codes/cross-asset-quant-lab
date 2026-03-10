from __future__ import annotations

from pathlib import Path
from typing import Optional
import pandas as pd

from src.data.registry import DataRegistry
from src.data.loaders.base import read_symbol_csv_dir
from src.data.schema import standardize_dataset_df


class CryptoracleLoader:
    """
    Cryptoracle NORMAL loader

    Supported path pattern:
    /cryptoracle_data/NORMAL/{metric}/{freq}/{symbol}/*.csv

    Example:
    /NORMAL/active_community_count/1h/BTCUSDT/*.csv
    """

    DEFAULT_METRICS = [
        "active_community_count",
        "effective_message_count",
        "interaction_event_count",
        "mention_count",
        "negative_sentiment_ratio",
        "positive_sentiment_ratio",
        "sentiment_breakout_signal",
        "sentiment_cum_deviation",
        "sentiment_momentum_zscore",
        "sentiment_price_divergence",
        "sentiment_spread",
        "unique_user_count",
    ]

    def __init__(self, registry: Optional[DataRegistry] = None) -> None:
        self.registry = registry or DataRegistry()

    @property
    def root(self) -> Path:
        return self.registry.cryptoracle_root

    def list_metrics(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted([p.name for p in self.root.iterdir() if p.is_dir()])

    def list_symbols(
        self,
        metric: str,
        freq: str = "1h",
    ) -> list[str]:
        metric_root = self.root / metric / freq
        if not metric_root.exists():
            return []
        return sorted([p.name for p in metric_root.iterdir() if p.is_dir()])

    def get_symbol_dir(
        self,
        metric: str,
        symbol: str,
        freq: str = "1h",
    ) -> Path:
        return self.root / metric / freq / symbol

    def exists(
        self,
        metric: str,
        symbol: str,
        freq: str = "1h",
    ) -> bool:
        return self.get_symbol_dir(metric, symbol, freq).exists()

    def load_metric(
        self,
        metric: str,
        symbol: str,
        freq: str = "1h",
        start_time: str | None = None,
        end_time: str | None = None,
        value_col_name: str | None = None,
    ) -> pd.DataFrame:
        """
        value_col_name:
            if provided, rename standardized 'value' column to this name;
            otherwise rename to metric.
        """
        symbol_dir = self.get_symbol_dir(metric=metric, symbol=symbol, freq=freq)
        raw = read_symbol_csv_dir(symbol_dir)
        if raw.empty:
            return pd.DataFrame(columns=["datetime", "symbol"])

        df = standardize_dataset_df(
            raw,
            source="cryptoracle",
            dataset=metric,
            symbol=symbol,
        )
        if df.empty:
            return df

        target_value_col = value_col_name or metric
        if "value" in df.columns and target_value_col != "value":
            df = df.rename(columns={"value": target_value_col})

        if start_time is not None:
            df = df[df["datetime"] >= pd.Timestamp(start_time)]
        if end_time is not None:
            df = df[df["datetime"] <= pd.Timestamp(end_time)]

        return df.sort_values(["datetime", "symbol"]).reset_index(drop=True)

    def load_many_symbols_one_metric(
        self,
        metric: str,
        symbols: list[str],
        freq: str = "1h",
        start_time: str | None = None,
        end_time: str | None = None,
        skip_missing: bool = True,
        value_col_name: str | None = None,
    ) -> pd.DataFrame:
        dfs: list[pd.DataFrame] = []

        for symbol in symbols:
            if skip_missing and not self.exists(metric, symbol, freq):
                continue

            df = self.load_metric(
                metric=metric,
                symbol=symbol,
                freq=freq,
                start_time=start_time,
                end_time=end_time,
                value_col_name=value_col_name,
            )
            if not df.empty:
                dfs.append(df)

        if not dfs:
            return pd.DataFrame(columns=["datetime", "symbol"])

        out = pd.concat(dfs, axis=0, ignore_index=True)
        out = out.sort_values(["datetime", "symbol"]).reset_index(drop=True)
        return out

    def load_one_symbol_many_metrics(
        self,
        symbol: str,
        metrics: list[str],
        freq: str = "1h",
        start_time: str | None = None,
        end_time: str | None = None,
    ) -> dict[str, pd.DataFrame]:
        out: dict[str, pd.DataFrame] = {}
        for metric in metrics:
            out[metric] = self.load_metric(
                metric=metric,
                symbol=symbol,
                freq=freq,
                start_time=start_time,
                end_time=end_time,
                value_col_name=metric,
            )
        return out

    def load_many_symbols_many_metrics(
        self,
        symbols: list[str],
        metrics: list[str] | None = None,
        freq: str = "1h",
        start_time: str | None = None,
        end_time: str | None = None,
        skip_missing: bool = True,
    ) -> dict[str, pd.DataFrame]:
        metrics = metrics or self.DEFAULT_METRICS
        out: dict[str, pd.DataFrame] = {}

        for metric in metrics:
            out[metric] = self.load_many_symbols_one_metric(
                metric=metric,
                symbols=symbols,
                freq=freq,
                start_time=start_time,
                end_time=end_time,
                skip_missing=skip_missing,
                value_col_name=metric,
            )
        return out