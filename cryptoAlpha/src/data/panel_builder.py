"""Panel construction utilities across multiple data sources."""
from __future__ import annotations

from typing import Iterable
import pandas as pd


KEY_COLS = ["datetime", "symbol"]


def _ensure_key_cols(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=KEY_COLS)

    out = df.copy()
    missing = [c for c in KEY_COLS if c not in out.columns]
    if missing:
        raise ValueError(f"{name} missing key columns: {missing}")

    out["datetime"] = (
        pd.to_datetime(out["datetime"], errors="coerce", utc=True)
        .dt.tz_localize(None)
    )
    out["symbol"] = out["symbol"].astype(str)
    out = out.sort_values(KEY_COLS).reset_index(drop=True)
    return out


def _deduplicate_on_keys(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if df.empty:
        return df

    dup_cnt = int(df.duplicated(subset=KEY_COLS).sum())
    if dup_cnt > 0:
        print(f"[WARN] {name} has {dup_cnt} duplicated rows on {KEY_COLS}, keep last.")
        df = df.drop_duplicates(subset=KEY_COLS, keep="last")

    return df.sort_values(KEY_COLS).reset_index(drop=True)


def _safe_merge(left: pd.DataFrame, right: pd.DataFrame, right_name: str) -> pd.DataFrame:
    if left.empty:
        return right.copy()
    if right.empty:
        return left.copy()

    overlap = [c for c in right.columns if c in left.columns and c not in KEY_COLS]
    if overlap:
        raise ValueError(
            f"Column overlap detected when merging {right_name}: {overlap}. "
            f"Please rename columns before merging."
        )

    out = left.merge(right, on=KEY_COLS, how="outer")
    out = out.sort_values(KEY_COLS).reset_index(drop=True)
    return out


def rename_columns(
    df: pd.DataFrame,
    rename_map: dict[str, str],
    name: str = "df",
) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=KEY_COLS)

    out = _ensure_key_cols(df, name)
    missing = [c for c in rename_map if c not in out.columns]
    if missing:
        raise ValueError(f"{name} missing columns for rename: {missing}")

    out = out.rename(columns=rename_map)
    out = _deduplicate_on_keys(out, name)
    return out


def merge_frames(frames: list[tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    """
    Parameters
    ----------
    frames:
        list of (name, dataframe), all dataframes must contain ['datetime', 'symbol']

    Returns
    -------
    panel_df:
        outer-merged panel
    """
    panel = pd.DataFrame(columns=KEY_COLS)

    for name, df in frames:
        df = _ensure_key_cols(df, name)
        df = _deduplicate_on_keys(df, name)
        panel = _safe_merge(panel, df, right_name=name)

    panel = panel.sort_values(KEY_COLS).reset_index(drop=True)
    return panel


def build_panel_from_dict(frame_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = [(name, df) for name, df in frame_dict.items()]
    return merge_frames(frames)


def build_base_panel(
    price_df: pd.DataFrame,
    funding_df: pd.DataFrame | None = None,
    oi_df: pd.DataFrame | None = None,
    taker_df: pd.DataFrame | None = None,
    ls_df: pd.DataFrame | None = None,
    gecko_df: pd.DataFrame | None = None,
    oracle_df_dict: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """
    一个适合你当前项目的基础 panel builder。

    约定：
    - price_df 已经标准化为:
      ['datetime','symbol','open','high','low','close','volume_usd']
    - funding_df:
      ['datetime','symbol','funding_open','funding_high','funding_low','funding_close']
    - oi_df:
      ['datetime','symbol','oi_open','oi_high','oi_low','oi_close']
    - taker_df:
      ['datetime','symbol','taker_buy_volume_usd','taker_sell_volume_usd']
    - ls_df:
      ['datetime','symbol','global_account_long_percent',
       'global_account_short_percent','global_account_long_short_ratio']
    - gecko_df:
      ['datetime','symbol','price','market_cap','total_volume']
    - oracle_df_dict:
      每个 df 至少为 ['datetime','symbol',metric_name]
    """
    frames: list[tuple[str, pd.DataFrame]] = [
        ("price", price_df),
    ]

    if funding_df is not None:
        frames.append(("funding", funding_df))
    if oi_df is not None:
        frames.append(("oi", oi_df))
    if taker_df is not None:
        frames.append(("taker", taker_df))
    if ls_df is not None:
        frames.append(("ls_ratio", ls_df))
    if gecko_df is not None:
        frames.append(("gecko", gecko_df))

    if oracle_df_dict:
        for metric_name, metric_df in oracle_df_dict.items():
            frames.append((f"oracle_{metric_name}", metric_df))

    panel = merge_frames(frames)
    return panel


def filter_panel_by_symbols(panel_df: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    df = _ensure_key_cols(panel_df, "panel_df")
    symbol_set = set(map(str, symbols))
    df = df[df["symbol"].isin(symbol_set)].copy()
    return df.sort_values(KEY_COLS).reset_index(drop=True)


def filter_panel_by_time(
    panel_df: pd.DataFrame,
    start_time: str | None = None,
    end_time: str | None = None,
) -> pd.DataFrame:
    df = _ensure_key_cols(panel_df, "panel_df")

    if start_time is not None:
        df = df[df["datetime"] >= pd.Timestamp(start_time)]
    if end_time is not None:
        df = df[df["datetime"] <= pd.Timestamp(end_time)]

    return df.sort_values(KEY_COLS).reset_index(drop=True)


def panel_info(panel_df: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_key_cols(panel_df, "panel_df")
    rows = []

    for col in df.columns:
        rows.append(
            {
                "column": col,
                "dtype": str(df[col].dtype),
                "non_null_ratio": float(df[col].notna().mean()),
                "n_unique": int(df[col].nunique(dropna=True)),
            }
        )

    info_df = pd.DataFrame(rows)
    return info_df


def make_price_panel(
    price_df: pd.DataFrame,
    gecko_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    如果你只想先做价格+市值相关因子，可以先用这个简版。
    """
    frames = [("price", price_df)]
    if gecko_df is not None:
        frames.append(("gecko", gecko_df))
    return merge_frames(frames)