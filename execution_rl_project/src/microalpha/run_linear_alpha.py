# filepath: src/microalpha/run_linear_alpha.py
from __future__ import annotations

from collections import deque
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.backtest.tardis_wrapper import TardisExecutionWrapper
from src.utils.io import load_yaml


FEATURE_COLS = [
    "spread_bps",
    "imb1",
    "imb3",
    "imb5",
    "microdev_bps",
    "depth_ratio_5",
    "ret_1_bps",
    "ret_5_bps",
    "ret_10_bps",
    "vol_10_bps",
    "ofi_1",
    "ofi_5",
    "trade_imb",
    "signed_vol_log",
]


def _safe_ratio(a: float, b: float) -> float:
    return a / (b + 1e-12)


def build_features_from_state(
    state,
    prev_mid_1: float | None,
    prev_mid_5: float | None,
    prev_mid_10: float | None,
    prev_bid1: float | None,
    prev_ask1: float | None,
    prev_bid1_size: float | None,
    prev_ask1_size: float | None,
    prev_bid5_sum: float | None,
    prev_ask5_sum: float | None,
    recent_mids: list[float],
) -> dict:
    best_bid = float(state.best_bid)
    best_ask = float(state.best_ask)
    mid = 0.5 * (best_bid + best_ask)

    bid_prices = np.asarray(state.bid_prices, dtype=float)
    ask_prices = np.asarray(state.ask_prices, dtype=float)
    bid_sizes = np.asarray(state.bid_sizes, dtype=float)
    ask_sizes = np.asarray(state.ask_sizes, dtype=float)

    spread_bps = _safe_ratio(best_ask - best_bid, mid) * 10000.0 if mid > 0 else 0.0

    bid1 = float(bid_sizes[0]) if len(bid_sizes) > 0 else 0.0
    ask1 = float(ask_sizes[0]) if len(ask_sizes) > 0 else 0.0

    bid3 = float(np.sum(bid_sizes[:3]))
    ask3 = float(np.sum(ask_sizes[:3]))
    bid5 = float(np.sum(bid_sizes[:5]))
    ask5 = float(np.sum(ask_sizes[:5]))

    imb1 = _safe_ratio(bid1 - ask1, bid1 + ask1)
    imb3 = _safe_ratio(bid3 - ask3, bid3 + ask3)
    imb5 = _safe_ratio(bid5 - ask5, bid5 + ask5)

    microprice = (best_ask * bid1 + best_bid * ask1) / (bid1 + ask1 + 1e-12)
    microdev_bps = _safe_ratio(microprice - mid, mid) * 10000.0 if mid > 0 else 0.0

    depth_ratio_5 = np.log((bid5 + 1e-12) / (ask5 + 1e-12))

    ret_1_bps = 0.0 if prev_mid_1 is None or prev_mid_1 <= 0 else _safe_ratio(mid - prev_mid_1, prev_mid_1) * 10000.0
    ret_5_bps = 0.0 if prev_mid_5 is None or prev_mid_5 <= 0 else _safe_ratio(mid - prev_mid_5, prev_mid_5) * 10000.0
    ret_10_bps = 0.0 if prev_mid_10 is None or prev_mid_10 <= 0 else _safe_ratio(mid - prev_mid_10, prev_mid_10) * 10000.0

    if len(recent_mids) >= 10:
        mids_arr = np.asarray(recent_mids[-10:], dtype=float)
        mid_rets = np.diff(mids_arr) / (mids_arr[:-1] + 1e-12)
        vol_10_bps = float(np.std(mid_rets) * 10000.0)
    else:
        vol_10_bps = 0.0

    # very lightweight approximate OFI
    if prev_bid1 is None or prev_ask1 is None or prev_bid1_size is None or prev_ask1_size is None:
        ofi_1 = 0.0
    else:
        bid_term = 0.0
        ask_term = 0.0

        if best_bid > prev_bid1:
            bid_term += bid1
        elif best_bid < prev_bid1:
            bid_term -= prev_bid1_size
        else:
            bid_term += (bid1 - prev_bid1_size)

        if best_ask < prev_ask1:
            ask_term += ask1
        elif best_ask > prev_ask1:
            ask_term -= prev_ask1_size
        else:
            ask_term += (ask1 - prev_ask1_size)

        ofi_1 = bid_term - ask_term

    if prev_bid5_sum is None or prev_ask5_sum is None:
        ofi_5 = 0.0
    else:
        ofi_5 = (bid5 - prev_bid5_sum) - (ask5 - prev_ask5_sum)

    signed_vol = float(state.recent_signed_volume)
    signed_vol_log = float(np.sign(signed_vol) * np.log1p(abs(signed_vol)))

    return {
        "timestamp_ns": int(state.timestamp_ns),
        "best_bid": best_bid,
        "best_ask": best_ask,
        "mid": mid,
        "spread_bps": spread_bps,
        "imb1": float(imb1),
        "imb3": float(imb3),
        "imb5": float(imb5),
        "microdev_bps": microdev_bps,
        "depth_ratio_5": float(depth_ratio_5),
        "ret_1_bps": ret_1_bps,
        "ret_5_bps": ret_5_bps,
        "ret_10_bps": ret_10_bps,
        "vol_10_bps": vol_10_bps,
        "ofi_1": float(ofi_1),
        "ofi_5": float(ofi_5),
        "trade_imb": float(state.recent_trade_imbalance),
        "signed_vol_log": signed_vol_log,
    }


def build_feature_table(
    wrapper: TardisExecutionWrapper,
    sample_interval_sec: float = 1.0,
    start_idx: int = 2000,
) -> pd.DataFrame:
    rows: list[dict] = []

    state = wrapper.reset(start_idx=start_idx)

    mid_hist: deque[float] = deque(maxlen=32)

    prev_bid1 = None
    prev_ask1 = None
    prev_bid1_size = None
    prev_ask1_size = None
    prev_bid5_sum = None
    prev_ask5_sum = None

    while True:
        current_mid = 0.5 * (state.best_bid + state.best_ask)

        prev_mid_1 = mid_hist[-1] if len(mid_hist) >= 1 else None
        prev_mid_5 = mid_hist[-5] if len(mid_hist) >= 5 else None
        prev_mid_10 = mid_hist[-10] if len(mid_hist) >= 10 else None

        feat = build_features_from_state(
            state=state,
            prev_mid_1=prev_mid_1,
            prev_mid_5=prev_mid_5,
            prev_mid_10=prev_mid_10,
            prev_bid1=prev_bid1,
            prev_ask1=prev_ask1,
            prev_bid1_size=prev_bid1_size,
            prev_ask1_size=prev_ask1_size,
            prev_bid5_sum=prev_bid5_sum,
            prev_ask5_sum=prev_ask5_sum,
            recent_mids=list(mid_hist),
        )
        rows.append(feat)

        mid_hist.append(current_mid)

        prev_bid1 = float(state.best_bid)
        prev_ask1 = float(state.best_ask)
        prev_bid1_size = float(state.bid_sizes[0])
        prev_ask1_size = float(state.ask_sizes[0])
        prev_bid5_sum = float(np.sum(state.bid_sizes[:5]))
        prev_ask5_sum = float(np.sum(state.ask_sizes[:5]))

        if wrapper.is_done():
            break

        wrapper.step_time(sample_interval_sec)

        if wrapper.is_done():
            break

        state = wrapper.get_market_state()

    df = pd.DataFrame(rows)
    if len(df) == 0:
        raise ValueError("feature table is empty")

    df["datetime"] = pd.to_datetime(df["timestamp_ns"], unit="ns", utc=True)
    df["date"] = df["datetime"].dt.strftime("%Y-%m-%d")
    return df


def add_future_return_label(
    df: pd.DataFrame,
    horizon_steps: int = 5,
) -> pd.DataFrame:
    out = df.copy()
    future_mid = out["mid"].shift(-horizon_steps)
    out["future_return_bps"] = ((future_mid - out["mid"]) / out["mid"]) * 10000.0
    out = out.dropna().reset_index(drop=True)
    return out


def split_by_day(
    df: pd.DataFrame,
    train_days: list[str],
    test_days: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = df[df["date"].isin(train_days)].copy()
    test_df = df[df["date"].isin(test_days)].copy()

    if len(train_df) == 0 or len(test_df) == 0:
        raise ValueError("train/test split by day produced empty dataframe")

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def fit_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_type: str = "ridge",
):
    X_train = train_df[FEATURE_COLS].to_numpy()
    y_train = train_df["future_return_bps"].to_numpy()

    X_test = test_df[FEATURE_COLS].to_numpy()

    if model_type == "ridge":
        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("reg", Ridge(alpha=3.0)),
            ]
        )
    elif model_type == "hgbt":
        model = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.05,
            max_depth=4,
            max_iter=200,
            min_samples_leaf=100,
            l2_regularization=1.0,
            random_state=42,
        )
    else:
        raise ValueError(f"unknown model_type: {model_type}")

    model.fit(X_train, y_train)
    pred_test = model.predict(X_test)
    return model, pred_test


def evaluate_predictions(
    pred: np.ndarray,
    y_true: np.ndarray,
) -> dict:
    pred = np.asarray(pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)

    if len(pred) != len(y_true):
        raise ValueError("pred and y_true length mismatch")

    corr = float(np.corrcoef(pred, y_true)[0, 1]) if len(pred) > 1 else np.nan
    mse = float(mean_squared_error(y_true, pred))
    mae = float(np.mean(np.abs(pred - y_true)))

    pred_sign = np.sign(pred)
    true_sign = np.sign(y_true)
    sign_acc = float(np.mean(pred_sign == true_sign))

    # 只在有方向的样本上看方向准确率
    mask = np.abs(y_true) > 1e-12
    dir_acc_nonzero = float(np.mean(pred_sign[mask] == true_sign[mask])) if np.any(mask) else np.nan

    return {
        "pearson_corr": corr,
        "mse": mse,
        "mae": mae,
        "sign_acc": sign_acc,
        "dir_acc_nonzero": dir_acc_nonzero,
    }


def print_model_coefficients(model) -> None:
    if not isinstance(model, Pipeline):
        print("\n=== Nonlinear model: no linear coefficients to print ===")
        return

    reg = model.named_steps["reg"]
    coef = reg.coef_

    coef_df = pd.DataFrame(
        {
            "feature": FEATURE_COLS,
            "coef": coef,
            "abs_coef": np.abs(coef),
        }
    ).sort_values("abs_coef", ascending=False)

    print("\n=== Linear Model Coefficients ===")
    print(coef_df[["feature", "coef"]].to_string(index=False))


def run_one_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    out_dir: Path,
    model_type: str,
) -> None:
    model, pred_test = fit_model(train_df, test_df, model_type=model_type)

    y_test = test_df["future_return_bps"].to_numpy()
    metrics = evaluate_predictions(pred_test, y_test)

    print(f"\n================ {model_type.upper()} ================")
    print("=== Evaluation Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")

    print_model_coefficients(model)

    pred_df = test_df.copy()
    pred_df["pred_future_return_bps"] = pred_test

    pred_csv_path = out_dir / f"test_predictions_{model_type}.csv"
    model_path = out_dir / f"{model_type}_alpha.pkl"

    pred_df.to_csv(pred_csv_path, index=False)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print("\n=== Saved Files ===")
    print(f"test predictions: {pred_csv_path}")
    print(f"model: {model_path}")


def main() -> None:
    env_cfg = load_yaml("configs/env.yaml")
    asset_cfg = env_cfg["asset"]
    bt_cfg = env_cfg["backtest"]

    book_path = "/home/joyce/projects/data/raw/tardis/BTCUSDT/incremental_book_L2/BTCUSDT_2025-12-05_2025-12-07.csv"
    trade_path = "/home/joyce/projects/data/raw/tardis/BTCUSDT/trades/BTCUSDT_2025-12-05_2025-12-07.csv"
    snapshot_path = "/home/joyce/projects/data/raw/tardis/BTCUSDT/snapshot_25/BTCUSDT_2025-12-05_2025-12-07.csv.gz"

    sample_interval_sec = 1.0
    horizon_sec = 5.0
    horizon_steps = int(round(horizon_sec / sample_interval_sec))

    train_days = ["2025-12-05", "2025-12-06"]
    test_days = ["2025-12-07"]

    out_dir = Path("results/microalpha_v2")
    out_dir.mkdir(parents=True, exist_ok=True)

    wrapper = TardisExecutionWrapper(
        book_path=book_path,
        trade_path=trade_path,
        snapshot_path=snapshot_path,
        symbol=asset_cfg["symbol"],
        maker_fee=bt_cfg["maker_fee"],
        taker_fee=bt_cfg["taker_fee"],
        tick_size=asset_cfg["tick_size"],
        roi_lb=bt_cfg["roi_lb"],
        roi_ub=bt_cfg["roi_ub"],
        top_k=5,
    )

    print("[microalpha] building feature table ...")
    feat_df = build_feature_table(
        wrapper=wrapper,
        sample_interval_sec=sample_interval_sec,
        start_idx=2000,
    )

    print(f"[microalpha] raw feature rows: {len(feat_df)}")

    df = add_future_return_label(
        df=feat_df,
        horizon_steps=horizon_steps,
    )

    print(f"[microalpha] labeled rows: {len(df)}")
    print(f"[microalpha] horizon_sec: {horizon_sec}")
    print(f"[microalpha] sample_interval_sec: {sample_interval_sec}")
    print(f"[microalpha] train_days: {train_days}")
    print(f"[microalpha] test_days: {test_days}")

    train_df, test_df = split_by_day(
        df=df,
        train_days=train_days,
        test_days=test_days,
    )

    print(f"[microalpha] train rows: {len(train_df)}")
    print(f"[microalpha] test rows: {len(test_df)}")

    full_csv_path = out_dir / "feature_table_v2.csv"
    df.to_csv(full_csv_path, index=False)
    print(f"[microalpha] saved feature table: {full_csv_path}")

    run_one_model(train_df, test_df, out_dir, model_type="ridge")
    run_one_model(train_df, test_df, out_dir, model_type="hgbt")


if __name__ == "__main__":
    main()