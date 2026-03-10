from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Optional
import json

import pandas as pd
import yaml


TIME_CANDIDATES = [
    "datetime", "date", "time", "timestamp", "ts", "open_time", "close_time", "t",
]
SYMBOL_CANDIDATES = [
    "symbol", "ticker", "instId", "instrument",
]


@dataclass
class FileProfile:
    source: str
    dataset: str
    symbol: str
    file_path: str
    n_rows: int
    n_cols: int
    columns: list[str]
    dtypes: dict[str, str]
    time_col: Optional[str]
    symbol_col: Optional[str]
    parseable_time_ratio: Optional[float]
    min_time: Optional[str]
    max_time: Optional[str]
    duplicated_time_rows: Optional[int]
    null_ratio_by_col: dict[str, float]
    numeric_cols: list[str]
    sample_head: list[dict]


def load_paths(config_path: str = "configs/paths.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def first_existing(cols: Iterable[str], candidates: list[str]) -> Optional[str]:
    cols_lower_map = {str(c).lower(): str(c) for c in cols}
    for c in candidates:
        if c.lower() in cols_lower_map:
            return cols_lower_map[c.lower()]
    return None


def safe_read_csv(file_path: Path, nrows: Optional[int] = None) -> pd.DataFrame:
    # 先默认 utf-8，不行再 fallback
    tried = []
    for encoding in ["utf-8", "utf-8-sig", "gbk", "latin1"]:
        try:
            return pd.read_csv(file_path, nrows=nrows, encoding=encoding)
        except Exception as e:
            tried.append((encoding, str(e)))
    raise RuntimeError(f"Failed to read {file_path}. Tried: {tried}")


def profile_one_csv(
    file_path: Path,
    source: str,
    dataset: str,
    symbol: str,
    preview_rows: int = 50000,
) -> FileProfile:
    df = safe_read_csv(file_path, nrows=preview_rows)
    df.columns = [str(c).strip() for c in df.columns]

    time_col = first_existing(df.columns, TIME_CANDIDATES)
    symbol_col = first_existing(df.columns, SYMBOL_CANDIDATES)

    parseable_time_ratio = None
    min_time = None
    max_time = None
    duplicated_time_rows = None

    if time_col is not None:
        dt = pd.to_datetime(df[time_col], errors="coerce", utc=False)
        parseable_time_ratio = float(dt.notna().mean()) if len(dt) > 0 else None
        if dt.notna().any():
            min_time = str(dt.min())
            max_time = str(dt.max())
            duplicated_time_rows = int(dt.duplicated().sum())

    null_ratio_by_col = {
        col: float(df[col].isna().mean()) for col in df.columns
    }

    numeric_cols = [
        col for col in df.columns
        if pd.api.types.is_numeric_dtype(df[col])
    ]

    sample_head = df.head(5).to_dict(orient="records")

    return FileProfile(
        source=source,
        dataset=dataset,
        symbol=symbol,
        file_path=str(file_path),
        n_rows=int(df.shape[0]),
        n_cols=int(df.shape[1]),
        columns=df.columns.tolist(),
        dtypes={col: str(dtype) for col, dtype in df.dtypes.items()},
        time_col=time_col,
        symbol_col=symbol_col,
        parseable_time_ratio=parseable_time_ratio,
        min_time=min_time,
        max_time=max_time,
        duplicated_time_rows=duplicated_time_rows,
        null_ratio_by_col=null_ratio_by_col,
        numeric_cols=numeric_cols,
        sample_head=sample_head,
    )


def list_symbol_dirs(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted([p for p in path.iterdir() if p.is_dir()])


def list_csv_files(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted(path.glob("*.csv"))


def sample_first_csv_in_symbol_dir(symbol_dir: Path) -> Optional[Path]:
    csvs = list_csv_files(symbol_dir)
    return csvs[0] if csvs else None


def profile_coinglass(
    cfg: dict,
    datasets: list[str],
    freq: str = "1h",
    exchange: str = "Binance",
    max_symbols_per_dataset: int = 5,
) -> list[FileProfile]:
    root = Path(cfg["sources"]["coinglass_root"])
    profiles: list[FileProfile] = []

    for dataset in datasets:
        ds_root = root / dataset / freq / exchange
        symbol_dirs = list_symbol_dirs(ds_root)[:max_symbols_per_dataset]

        for symbol_dir in symbol_dirs:
            file_path = sample_first_csv_in_symbol_dir(symbol_dir)
            if file_path is None:
                continue

            profile = profile_one_csv(
                file_path=file_path,
                source="coinglass",
                dataset=dataset,
                symbol=symbol_dir.name,
            )
            profiles.append(profile)

    return profiles


def profile_coingecko(
    cfg: dict,
    dataset: str = "COIN_MARKET_CHART/hourly",
    max_symbols: int = 5,
) -> list[FileProfile]:
    root = Path(cfg["sources"]["coingecko_root"]) / dataset
    profiles: list[FileProfile] = []

    symbol_dirs = list_symbol_dirs(root)[:max_symbols]
    for symbol_dir in symbol_dirs:
        file_path = sample_first_csv_in_symbol_dir(symbol_dir)
        if file_path is None:
            continue

        profile = profile_one_csv(
            file_path=file_path,
            source="coingecko",
            dataset=dataset,
            symbol=symbol_dir.name,
        )
        profiles.append(profile)

    return profiles


def profile_cryptoracle(
    cfg: dict,
    metrics: list[str],
    freq: str = "1h",
    max_symbols_per_metric: int = 5,
) -> list[FileProfile]:
    root = Path(cfg["sources"]["cryptoracle_root"])
    profiles: list[FileProfile] = []

    for metric in metrics:
        metric_root = root / metric / freq
        symbol_dirs = list_symbol_dirs(metric_root)[:max_symbols_per_metric]

        for symbol_dir in symbol_dirs:
            file_path = sample_first_csv_in_symbol_dir(symbol_dir)
            if file_path is None:
                continue

            profile = profile_one_csv(
                file_path=file_path,
                source="cryptoracle",
                dataset=metric,
                symbol=symbol_dir.name,
            )
            profiles.append(profile)

    return profiles


def build_schema_summary(profiles: list[FileProfile]) -> pd.DataFrame:
    rows = []
    for p in profiles:
        rows.append(
            {
                "source": p.source,
                "dataset": p.dataset,
                "symbol": p.symbol,
                "file_path": p.file_path,
                "n_rows": p.n_rows,
                "n_cols": p.n_cols,
                "time_col": p.time_col,
                "symbol_col": p.symbol_col,
                "parseable_time_ratio": p.parseable_time_ratio,
                "min_time": p.min_time,
                "max_time": p.max_time,
                "duplicated_time_rows": p.duplicated_time_rows,
                "columns": " | ".join(p.columns),
                "numeric_cols": " | ".join(p.numeric_cols),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    cfg = load_paths("configs/paths.yaml")
    out_dir = Path("data/cache/eda")
    out_dir.mkdir(parents=True, exist_ok=True)

    coinglass_profiles = profile_coinglass(
        cfg,
        datasets=[
            "FUTURES_PRICE_HISTORY",
            "FUTURES_FUNDING_RATE_HISTORY",
            "FUTURES_OPEN_INTEREST",
            "FUTURES_TAKER_BUY_SELL_VOLUME",
            "FUTURES_GLOBAL_LS_ACCOUNT_RATIO",
        ],
        freq="1h",
        exchange="Binance",
        max_symbols_per_dataset=5,
    )

    coingecko_profiles = profile_coingecko(
        cfg,
        dataset="COIN_MARKET_CHART/hourly",
        max_symbols=5,
    )

    cryptoracle_profiles = profile_cryptoracle(
        cfg,
        metrics=[
            "active_community_count",
            # 你后面可继续加
            # "positive_sentiment_ratio",
            # "negative_sentiment_ratio",
        ],
        freq="1h",
        max_symbols_per_metric=5,
    )

    all_profiles = coinglass_profiles + coingecko_profiles + cryptoracle_profiles

    # 明细 JSON
    json_fp = out_dir / "schema_profiles.json"
    with open(json_fp, "w", encoding="utf-8") as f:
        json.dump([asdict(p) for p in all_profiles], f, ensure_ascii=False, indent=2)

    # 汇总 CSV
    summary_df = build_schema_summary(all_profiles)
    csv_fp = out_dir / "schema_summary.csv"
    summary_df.to_csv(csv_fp, index=False)

    print(f"[INFO] saved profile json to: {json_fp}")
    print(f"[INFO] saved summary csv to: {csv_fp}")
    print()
    print(summary_df.to_string(max_colwidth=120))


if __name__ == "__main__":
    main()