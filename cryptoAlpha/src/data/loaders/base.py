from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


CSV_ENCODINGS = ["utf-8", "utf-8-sig", "gbk", "latin1"]


def find_csv_files(symbol_dir: Path) -> list[Path]:
    """
    返回目录下所有 csv，按文件名排序。
    """
    if not symbol_dir.exists() or not symbol_dir.is_dir():
        return []
    return sorted(symbol_dir.glob("*.csv"))


def read_one_csv(csv_path: Path) -> pd.DataFrame:
    """
    读取单个 csv，自动尝试多种编码。
    """
    last_err: Exception | None = None

    for encoding in CSV_ENCODINGS:
        try:
            return pd.read_csv(csv_path, encoding=encoding)
        except Exception as e:
            last_err = e

    raise RuntimeError(f"Failed to read csv: {csv_path}, last_err={last_err}")


def read_symbol_csv_dir(
    symbol_dir: Path,
    sort_files: bool = True,
    add_source_file: bool = False,
) -> pd.DataFrame:
    """
    读取一个 symbol 目录下的所有 csv 并纵向拼接。

    Parameters
    ----------
    symbol_dir : Path
        例如 .../BTCUSDT
    sort_files : bool
        是否对文件名排序后再读取
    add_source_file : bool
        是否给每行加一个 __source_file 列，便于排查问题
    """
    if not symbol_dir.exists() or not symbol_dir.is_dir():
        return pd.DataFrame()

    csv_files = list(symbol_dir.glob("*.csv"))
    if sort_files:
        csv_files = sorted(csv_files)

    if not csv_files:
        return pd.DataFrame()

    dfs: list[pd.DataFrame] = []
    for fp in csv_files:
        try:
            df = read_one_csv(fp)
            if add_source_file:
                df["__source_file"] = fp.name
            dfs.append(df)
        except Exception as e:
            print(f"[WARN] Failed reading {fp}: {e}")

    if not dfs:
        return pd.DataFrame()

    out = pd.concat(dfs, axis=0, ignore_index=True, sort=False)
    return out


def list_subdirs(root: Path) -> list[Path]:
    """
    返回 root 下所有一级子目录。
    """
    if not root.exists() or not root.is_dir():
        return []
    return sorted([p for p in root.iterdir() if p.is_dir()])


def list_subdir_names(root: Path) -> list[str]:
    """
    返回 root 下所有一级子目录名称。
    """
    return [p.name for p in list_subdirs(root)]