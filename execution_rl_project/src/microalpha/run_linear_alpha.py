# filepath: src/microalpha/run_linear_alpha.py
from __future__ import annotations

from collections import deque
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.backtest.tardis_wrapper import TardisExecutionWrapper
from src.utils.io import load_yaml


FEATURE_COLS = [
    "spread_bps",
    "imb1",
    "imb5",
    "microdev",
    "ret_1",
    "ret_5",
    "trade_imb",
    "signed_vol",
]


def build_features_from_state(
    state,
    prev_mid_1: float | None,
    prev_mid_5: float | None,
) -> dict:
    best_bid = float(state.best_bid)
    best_ask = float(state.best_ask)
    mid = 0.5 * (best_bid + best_ask)

    bid_sizes = np.asarray(state.bid_sizes, dtype=float)
    ask_sizes = np.asarray(state.ask_sizes, dtype=float)

    spread_bps = (best_ask - best_bid) / mid if mid > 0 else 0.0

    bid1 = float(bid_sizes[0]) if len(bid_sizes) > 0 else 0.0
    ask1 = float(ask_sizes[0]) if len(ask_sizes) > 0 else 0.0
    imb1 = (bid1 - ask1) / (bid1 + ask1 + 1e-12)

    bid5 = float(np.sum(bid_sizes[:5]))
    ask5 = float(np.sum(ask_sizes[:5]))
    imb5 = (bid5 - ask5) / (bid5 + ask5 + 1e-12)

    # microprice
    microprice = (
        best_ask * bid1 + best_bid * ask1
    ) / (bid1 + ask1 + 1e-12)
    microdev = (microprice - mid) / mid if mid > 0 else 0.0

    ret_1 = 0.0 if prev_mid_1 is None or prev_mid_1 <= 0 else (mid - prev_mid_1) / prev_mid_1
    ret_5 = 0.0 if prev_mid_5 is None or prev_mid_5 <= 0 else (mid - prev_mid_5) / prev_mid_5

    return {
        "timestamp_ns": int(state.timestamp_ns),
        "best_bid": best_bid,
        "best_ask": best_ask,
        "mid": mid,
        "spread_bps": float(spread_bps),
        "imb1": float(imb1),
        "imb5": float(imb5),
        "microdev": float(microdev),
        "ret_1": float(ret_1),
        "ret_5": float(ret_5),
        "trade_imb": float(state.recent_trade_imbalance),
        "signed_vol": float(state.recent_signed_volume),
    }


def build_feature_table(
    wrapper: TardisExecutionWrapper,
    sample_interval_sec: float = 1.0,
    start_idx: int = 0,
) -> pd.DataFrame:
    rows: list[dict] = []

    # reset replay to start
    state = wrapper.reset(start_idx=start_idx)

    # keep recent mids to build lagged returns
    mid_hist: deque[float] = deque(maxlen=10)

    while True:
        current_mid = 0.5 * (state.best_bid + state.best_ask)
        prev_mid_1 = mid_hist[-1] if len(mid_hist) >= 1 else None
        prev_mid_5 = mid_hist[-5] if len(mid_hist) >= 5 else None

        feat = build_features_from_state(
            state=state,
            prev_mid_1=prev_mid_1,
            prev_mid_5=prev_mid_5,
        )
        rows.append(feat)

        mid_hist.append(current_mid)

        if wrapper.is_done():
            break

        wrapper.step_time(sample_interval_sec)

        if wrapper.is_done():
            break

        state = wrapper.get_market_state()

    df = pd.DataFrame(rows)
    if len(df) == 0:
        raise ValueError("feature table is empty")
    return df


def add_future_return_label(
    df: pd.DataFrame,
    horizon_steps: int = 5,
) -> pd.DataFrame:
    out = df.copy()
    future_mid = out["mid"].shift(-horizon_steps)
    out["future_return"] = (future_mid - out["mid"]) / out["mid"]
    out = out.dropna().reset_index(drop=True)
    return out


def time_split_train_test(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio must be between 0 and 1")

    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    if len(train_df) == 0 or len(test_df) == 0:
        raise ValueError("train/test split produced empty dataframe")

    return train_df, test_df


def fit_linear_alpha(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[Pipeline, np.ndarray]:
    X_train = train_df[FEATURE_COLS].to_numpy()
    y_train = train_df["future_return"].to_numpy()

    X_test = test_df[FEATURE_COLS].to_numpy()

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ]
    )
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

    # directional accuracy only on non-flat true labels
    mask = true_sign != 0
    dir_acc_nonzero = float(np.mean(pred_sign[mask] == true_sign[mask])) if np.any(mask) else np.nan

    return {
        "pearson_corr": corr,
        "mse": mse,
        "mae": mae,
        "sign_acc": sign_acc,
        "dir_acc_nonzero": dir_acc_nonzero,
    }


def print_model_coefficients(model: Pipeline) -> None:
    ridge: Ridge = model.named_steps["ridge"]
    coef = ridge.coef_

    coef_df = pd.DataFrame(
        {
            "feature": FEATURE_COLS,
            "coef": coef,
            "abs_coef": np.abs(coef),
        }
    ).sort_values("abs_coef", ascending=False)

    print("\n=== Linear Model Coefficients ===")
    print(coef_df[["feature", "coef"]].to_string(index=False))


def main() -> None:
    # paths
    env_cfg = load_yaml("configs/env.yaml")
    asset_cfg = env_cfg["asset"]
    bt_cfg = env_cfg["backtest"]

    book_path = "/home/joyce/test.csv"
    trade_path = "/home/joyce/test_trades.csv"
    snapshot_path = "/home/joyce/projects/data/raw/tardis/BTCUSDT/snapshot_25/test_book.csv"

    sample_interval_sec = 1.0
    horizon_sec = 5.0
    horizon_steps = int(round(horizon_sec / sample_interval_sec))

    out_dir = Path("results/microalpha")
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
        start_idx=0,
    )

    print(f"[microalpha] raw feature rows: {len(feat_df)}")

    df = add_future_return_label(
        df=feat_df,
        horizon_steps=horizon_steps,
    )

    print(f"[microalpha] labeled rows: {len(df)}")
    print(f"[microalpha] horizon_sec: {horizon_sec}")
    print(f"[microalpha] sample_interval_sec: {sample_interval_sec}")

    train_df, test_df = time_split_train_test(df, train_ratio=0.7)

    print(f"[microalpha] train rows: {len(train_df)}")
    print(f"[microalpha] test rows: {len(test_df)}")

    model, pred_test = fit_linear_alpha(train_df, test_df)

    y_test = test_df["future_return"].to_numpy()
    metrics = evaluate_predictions(pred_test, y_test)

    print("\n=== Evaluation Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")

    print_model_coefficients(model)

    # save outputs
    full_csv_path = out_dir / "feature_table.csv"
    pred_csv_path = out_dir / "test_predictions.csv"
    model_path = out_dir / "ridge_alpha.pkl"

    df.to_csv(full_csv_path, index=False)

    pred_df = test_df.copy()
    pred_df["pred_future_return"] = pred_test
    pred_df.to_csv(pred_csv_path, index=False)

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print("\n=== Saved Files ===")
    print(f"feature table: {full_csv_path}")
    print(f"test predictions: {pred_csv_path}")
    print(f"model: {model_path}")


if __name__ == "__main__":
    main()