from __future__ import annotations

from pathlib import Path
import boto3
import pandas as pd
from botocore.exceptions import ClientError

BUCKET = "quantbase-research-prod"
S3_ROOT = "2_TDS/binance-futures"

LOCAL_ROOT = Path("/home/joyce/projects/data/raw/tardis")

SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "SOLUSDT",
    "DOGEUSDT",
    "ADAUSDT",
    "TRXUSDT",
    "LINKUSDT",
    "LTCUSDT",
    "AVAXUSDT",
    "MATICUSDT",
    "DOTUSDT",
    "BCHUSDT",
    "XLMUSDT",
    "ATOMUSDT",
    "NEARUSDT",
    "ETCUSDT",
    "FILUSDT",
    "APTUSDT",
]

START_DAY = "2025-12-01"
END_DAY = "2026-01-01"

DATA_TYPES = {
    "incremental_book_L2": ".csv.gz",
    "trades": ".csv.gz",
    "book_snapshot_25": ".csv.gz",
}

s3 = boto3.client("s3")


def make_day_list(start_day: str, end_day: str) -> list[str]:
    dates = pd.date_range(start=start_day, end=end_day, freq="D")
    return [d.strftime("%Y-%m-%d") for d in dates]


def s3_key(data_type: str, symbol: str, day: str) -> str:
    ext = DATA_TYPES[data_type]
    return f"{S3_ROOT}/{data_type}/{symbol}/{day}/{symbol}{ext}"


def local_path(data_type: str, symbol: str, day: str) -> Path:
    if data_type == "book_snapshot_25":
        subdir = "snapshot_25"
        fname = f"{symbol}.csv.gz"
    else:
        subdir = data_type
        fname = f"{symbol}.csv"

    return LOCAL_ROOT / symbol / subdir / day / fname


def file_exists_on_s3(bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in {"404", "NoSuchKey", "NotFound"}:
            return False
        raise


def download_one(data_type: str, symbol: str, day: str, overwrite: bool = False) -> bool:
    key = s3_key(data_type, symbol, day)
    dst = local_path(data_type, symbol, day)

    if dst.exists() and not overwrite:
        print(f"[skip-local] {dst}")
        return True

    if not file_exists_on_s3(BUCKET, key):
        print(f"[missing-s3] s3://{BUCKET}/{key}")
        return False

    dst.parent.mkdir(parents=True, exist_ok=True)
    s3.download_file(BUCKET, key, str(dst))
    print(f"[downloaded] s3://{BUCKET}/{key} -> {dst}")
    return True


def main() -> None:
    days = make_day_list(START_DAY, END_DAY)

    print("=== Download Tardis Top20 ===")
    print(f"bucket={BUCKET}")
    print(f"s3_root={S3_ROOT}")
    print(f"local_root={LOCAL_ROOT}")
    print(f"num_symbols={len(SYMBOLS)}")
    print(f"num_days={len(days)}")
    print(f"start={days[0]}")
    print(f"end={days[-1]}")
    print()

    total = 0
    ok = 0
    missing = 0

    for symbol in SYMBOLS:
        print(f"\n===== {symbol} =====")
        for day in days:
            for data_type in DATA_TYPES:
                total += 1
                success = download_one(
                    data_type=data_type,
                    symbol=symbol,
                    day=day,
                    overwrite=False,
                )
                if success:
                    ok += 1
                else:
                    missing += 1

    print("\n=== Summary ===")
    print(f"total requests: {total}")
    print(f"downloaded / exists: {ok}")
    print(f"missing on s3: {missing}")


if __name__ == "__main__":
    main()