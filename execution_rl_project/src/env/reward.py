from __future__ import annotations


def compute_step_reward(
    arrival_price: float,
    filled_qty: float,
    avg_fill_price: float,
    target_qty: float,
) -> tuple[float, dict]:
    reward = 0.0
    shortfall = 0.0

    if filled_qty > 0:
        shortfall = (avg_fill_price - arrival_price) / arrival_price
        reward = -10000 * shortfall

    parts = {
        "shortfall": float(shortfall),
        "shortfall_reward": float(reward),
    }
    return float(reward), parts


def compute_terminal_penalty(
    remaining_qty: float,
    target_qty: float,
    lambda_terminal_remain: float,
) -> float:
    remaining_ratio = max(0.0, remaining_qty) / max(target_qty, 1e-12)
    return -lambda_terminal_remain * remaining_ratio