from __future__ import annotations

import math

import numpy as np


def compute_lob_state_features(
    best_bid: float,
    best_ask: float,
    bid_prices: list[float],
    bid_sizes: list[float],
    ask_prices: list[float],
    ask_sizes: list[float],
    mid_history: list[float],
) -> np.ndarray:
    mid = 0.5 * (best_bid + best_ask)
    mid = max(mid, 1e-12)

    spread_bps = (best_ask - best_bid) / mid * 10000.0

    depth_bid10 = float(sum(bid_sizes[:10]))
    depth_ask10 = float(sum(ask_sizes[:10]))

    denom = depth_bid10 + depth_ask10
    imbalance10 = 0.0 if denom <= 1e-12 else (depth_bid10 - depth_ask10) / denom

    ask_prices_10 = ask_prices[:10]
    ask_sizes_10 = ask_sizes[:10]
    ask_size_sum = float(sum(ask_sizes_10))
    if ask_size_sum <= 1e-12:
        ask_vwap = best_ask
    else:
        ask_vwap = float(
            sum(p * q for p, q in zip(ask_prices_10, ask_sizes_10)) / ask_size_sum
        )
    liquidity_cost_ask10 = ask_vwap - mid

    volatility = 0.0
    if len(mid_history) >= 2:
        mids = np.asarray(mid_history, dtype=np.float64)
        mids = np.maximum(mids, 1e-12)
        log_ret = np.diff(np.log(mids))
        if log_ret.size > 0:
            volatility = float(np.sqrt(np.mean(log_ret**2)))

    return np.asarray(
        [
            spread_bps,
            depth_bid10,
            depth_ask10,
            imbalance10,
            liquidity_cost_ask10,
            volatility,
        ],
        dtype=np.float32,
    )


def build_execution_state_features(
    target_qty: float,
    filled_qty: float,
    current_step: int,
    max_steps: int,
) -> np.ndarray:
    remaining_inventory_ratio = max(0.0, target_qty - filled_qty) / max(target_qty, 1e-12)
    time_remaining_ratio = max(0.0, max_steps - current_step) / max(max_steps, 1)

    return np.asarray(
        [
            time_remaining_ratio,
            remaining_inventory_ratio,
        ],
        dtype=np.float32,
    )