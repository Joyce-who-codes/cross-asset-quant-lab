from pathlib import Path
import numpy as np


EVENT_DTYPE = np.dtype(
    [
        ("ev", "u8"),
        ("exch_ts", "i8"),
        ("local_ts", "i8"),
        ("px", "f8"),
        ("qty", "f8"),
        ("order_id", "u8"),
        ("ival", "i8"),
        ("fval", "f8"),
    ],
    align=True,
)


def build_dummy_events(n: int = 10000) -> np.ndarray:
    arr = np.zeros(n, dtype=EVENT_DTYPE)
    base_ts = 1_700_000_000_000_000_000
    px = 50000.0
    for i in range(n):
        arr[i]["ev"] = 1
        arr[i]["exch_ts"] = base_ts + i * 1_000_000
        arr[i]["local_ts"] = base_ts + i * 1_000_000 + 500_000
        px += np.random.randn() * 0.5
        arr[i]["px"] = px
        arr[i]["qty"] = max(0.001, abs(np.random.randn()) * 0.01)
        arr[i]["order_id"] = i
        arr[i]["ival"] = 0
        arr[i]["fval"] = 0.0
    return arr


def main() -> None:
    out_dir = Path("data/processed/hbt")
    out_dir.mkdir(parents=True, exist_ok=True)
    arr = build_dummy_events()
    np.savez_compressed(out_dir / "btcusdt_day1.npz", data=arr)
    np.savez_compressed(out_dir / "btcusdt_day2.npz", data=arr)


if __name__ == "__main__":
    main()