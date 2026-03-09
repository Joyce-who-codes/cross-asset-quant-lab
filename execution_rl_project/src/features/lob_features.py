from __future__ import annotations

import numpy as np


def safe_div(a: float, b: float) -> float:
    return a / b if abs(b) > 1e-12 else 0.0


def compute_basic_lob_features(
    best_bid: float,
    best_ask: float,
    bid_prices: list[float],
    bid_sizes: list[float],
    ask_prices: list[float],
    ask_sizes: list[float],
    recent_trade_imbalance: float = 0.0,
    recent_signed_volume: float = 0.0,
) -> np.ndarray:
    mid = 0.5 * (best_bid + best_ask)
    spread = max(0.0, best_ask - best_bid)

    bid_sum = float(np.sum(bid_sizes))
    ask_sum = float(np.sum(ask_sizes))
    imbalance = safe_div(bid_sum - ask_sum, bid_sum + ask_sum)

    feats = np.array(
        [
            best_bid,
            best_ask,
            mid,
            spread,
            *bid_prices[:5],
            *bid_sizes[:5],
            *ask_prices[:5],
            *ask_sizes[:5],
            bid_sum,
            ask_sum,
            imbalance,
            recent_trade_imbalance,
            recent_signed_volume,
        ],
        dtype=np.float32,
    )
    return feats


def build_execution_state_features(
    target_qty: float,
    filled_qty: float,
    remaining_steps: int,
    has_active_order: bool,
    active_order_price_offset: float,
) -> np.ndarray:
    remain_qty = max(0.0, target_qty - filled_qty)
    return np.array(
        [
            target_qty,
            filled_qty,
            remain_qty,
            remaining_steps,
            float(has_active_order),
            active_order_price_offset,
        ],
        dtype=np.float32,
    )